import pytest
import torch

from hla_tok import HLATok
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint
from hla_tok.training.losses import Losses
from hla_tok.training.step import initialize_codebook, optimize_batch


@pytest.fixture(autouse=True)
def configure_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def test_checkpoint_resume_replays_next_update_and_rejects_identity_changes(tmp_path):
    torch.manual_seed(23)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    budget_rng = torch.Generator().manual_seed(29)
    identity = ResumeIdentity("fixed", "experiment", "manifest", "commit")

    def step(layer, opt, lr, rng):
        opt.zero_grad(set_to_none=True)
        values = torch.randn(5, 4)
        factor = torch.rand((), generator=rng)
        (layer(values).square().mean() * factor).backward()
        opt.step()
        lr.step()

    step(model, optimizer, scheduler, budget_rng)
    path = tmp_path / "step_00001.pt"
    save_checkpoint(path, identity, model, optimizer, scheduler, budget_rng, step=1, data_cursor=1, counters={})
    step(model, optimizer, scheduler, budget_rng)
    replay = torch.nn.Linear(4, 2)
    replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=0.03)
    replay_lr = torch.optim.lr_scheduler.StepLR(replay_optimizer, step_size=2, gamma=0.9)
    replay_rng = torch.Generator()
    state = read_checkpoint(path, identity).restore(replay, replay_optimizer, replay_lr, replay_rng)
    assert state["data_cursor"] == 1 and state["step"] == 1
    step(replay, replay_optimizer, replay_lr, replay_rng)
    assert all(torch.equal(parameter, actual) for parameter, actual in zip(model.parameters(), replay.parameters()))
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, ResumeIdentity("fixed", "other", "manifest", "commit"))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, identity, model, optimizer, scheduler, budget_rng, step=1, data_cursor=1, counters={})


def test_codebook_initialization_is_repeatable_and_train_step_logs_actual_gradients():
    torch.manual_seed(27)
    model = HLATok()
    source = torch.rand(1, 3, 128, 128)
    book = model.quantizer
    # Fit a small test book with real latent vectors, then use the production book for the update.
    tiny = type(book)(size=8, dim=32)
    observations = model.encode_training(source, 31, quantize=False).values.detach().flatten(0, 1)
    initialize_codebook(tiny, observations, seed=7, iterations=2)
    expected = tiny.embedding.weight.detach().clone()
    initialize_codebook(tiny, observations, seed=7, iterations=2)
    assert torch.equal(tiny.embedding.weight, expected)
    assert torch.isfinite(expected).all()

    losses = Losses("cpu", lpips_vgg=0.0, dists=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    row = optimize_batch(model, source, (11, 31), optimizer, losses, audit=True)
    assert row["local_budgets"] == [11, 31]
    commitment = row["losses"]["vq_commitment"]
    assert commitment["weighted"] == pytest.approx(0.25 * commitment["raw"])
    assert commitment["gradient_l2"]["query_encoder"] > 0
    assert commitment["gradient_l2"]["quantizer"] == 0
    assert row["losses"]["vq_codebook"]["gradient_l2"]["quantizer"] > 0
    assert all(value > 0 for value in row["module_gradient_l2"].values())


def test_quality_gate_interpolates_observed_bpp_and_rejects_extrapolation():
    from hla_tok.training.train import fixed_quality_gate

    common = {"method": "Original AdapTok", "stride": 128, "precision": "fp32"}
    baseline = [
        common | {"actual_bpp": 0.06, "global_psnr": 24, "lpips": 0.25, "dists": 0.09},
        common | {"actual_bpp": 0.10, "global_psnr": 26, "lpips": 0.19, "dists": 0.07},
    ]
    measured = {"actual_bpp": 0.08, "global_psnr": 24, "lpips": 0.239, "dists": 0.10}
    record = fixed_quality_gate(measured, baseline)
    assert record["linear_bpp_interpolation"] == pytest.approx({"global_psnr": 25, "lpips": 0.22, "dists": 0.08})
    assert record["passes"] == {"psnr": True, "lpips": True, "dists": False}
    assert not record["quality_pass"]
    with pytest.raises(ValueError, match="bracket"):
        fixed_quality_gate(measured | {"actual_bpp": 0.04}, baseline)


def test_resume_accepts_report_commit_but_rejects_a_changed_codec(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    rng = torch.Generator().manual_seed(1)
    initial = ResumeIdentity("fixed", "config", "dataset", "run-commit", "code")
    report_only = ResumeIdentity("fixed", "config", "dataset", "report-commit", "code")
    incompatible = ResumeIdentity("fixed", "config", "dataset", "another-commit", "changed-code")
    path = tmp_path / "step_00001.pt"
    save_checkpoint(path, initial, model, optimizer, scheduler, rng, step=1, data_cursor=1, counters={})
    assert read_checkpoint(path, report_only).state["identity"]["git_commit"] == "run-commit"
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, incompatible)
