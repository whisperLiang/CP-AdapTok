import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F


@pytest.mark.parametrize("budget", (256, 512, 1024, 2048))
def test_active_vq_loss_and_gradient_are_exact_prefix(budget):
    from adaptok_hr.models.vq import VectorQuantizer

    vq = VectorQuantizer(loss_scope="active_prefix")
    values = torch.randn(1, 2685, 16, requires_grad=True)
    encoded = vq(values)
    terms = encoded.loss_terms(budget)
    z = F.normalize(values, dim=-1)[:, :budget]
    e = vq.lookup(encoded.ids[:, :budget])
    torch.testing.assert_close(terms["codebook_loss"], (z.detach() - e).square().mean())
    torch.testing.assert_close(terms["commitment_loss"], (z - e.detach()).square().mean())
    torch.testing.assert_close(terms["vq_loss"], terms["codebook_loss"] + .25 * terms["commitment_loss"])
    terms["vq_loss"].backward()
    assert values.grad[:, :budget].abs().sum() > 0
    assert values.grad[:, budget:].eq(0).all()
    assert vq.codebook.weight.grad.abs().sum() > 0
    active = torch.zeros(8192, dtype=torch.bool)
    active[encoded.ids[:, :budget].flatten()] = True
    assert vq.codebook.weight.grad[~active].eq(0).all()
    torch.testing.assert_close(encoded.quantized, vq.lookup(encoded.ids), atol=0, rtol=0)


def test_legacy_scope_still_supervises_all_candidates():
    from adaptok_hr.models.vq import VectorQuantizer

    values = torch.randn(1, 42, 16, requires_grad=True)
    encoded = VectorQuantizer(loss_scope="all_candidates")(values)
    encoded.loss_terms(4)["vq_loss"].backward()
    assert values.grad[:, 4:].abs().sum() > 0


def test_reconstruction_ste_reaches_only_active_tokens_and_never_codebook():
    from adaptok_hr.models.vq import VectorQuantizer

    values = torch.randn(1, 42, 16, requires_grad=True)
    vq = VectorQuantizer(loss_scope="active_prefix")
    encoded = vq(values)
    encoded.quantized[:, :16].sum().backward()
    assert values.grad is not None
    assert values.grad[:, :16].abs().sum() > 0 and values.grad[:, 16:].eq(0).all()
    assert all(p.grad is None or p.grad.eq(0).all() for p in vq.parameters())


def test_per_level_lookup_uses_position_not_id_and_hashes_every_book(small_config):
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.models.model import AdapTokHR

    model = AdapTokHR(replace(small_config, codebook_mode="per_level"))
    for level, book in enumerate(model.vq.books):
        with torch.no_grad():
            book.weight.zero_()
            book.weight[:, level] = 1
    ids = torch.zeros(1, 32, dtype=torch.long)
    values = model.vq.lookup(ids)
    for level, count in enumerate(model.positions.counts):
        offset = sum(model.positions.counts[:level])
        expected = torch.zeros(16)
        expected[level] = 1
        torch.testing.assert_close(values[:, offset:min(offset + count, 32)],
                                   expected.expand(1, min(count, max(0, 32 - offset)), 16))
    original = Codec(model.eval()).codebook_identity
    with torch.no_grad():
        model.vq.books[2].weight[0, 0] += .01
    assert original != Codec(model.eval()).codebook_identity


def test_codebook_count_does_not_change_encoder_decoder_initialization(small_config):
    from adaptok_hr.models.model import AdapTokHR

    torch.manual_seed(40)
    shared = AdapTokHR(small_config)
    torch.manual_seed(40)
    layered = AdapTokHR(replace(small_config, codebook_mode="per_level"))
    for name in ("encoder", "decoder", "tokenizer"):
        for key, value in getattr(shared, name).state_dict().items():
            torch.testing.assert_close(value, getattr(layered, name).state_dict()[key], atol=0, rtol=0)


def test_nested_vq_matches_reconstruction_weights_and_has_no_tail_gradient(small_config):
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.training.losses import ReconstructionLoss
    from adaptok_hr.training.train import frame_objective

    model = AdapTokHR(replace(small_config, loss_scope="active_prefix"))
    frame = torch.rand(1, 3, 128, 256)
    encoded = model.encode_features(frame)
    encoded.continuous.retain_grad()
    model.encode_features = lambda *args: encoded
    losses = ReconstructionLoss({"vq": 1.}, "cpu")
    loss, state, row = frame_objective(model, frame, None, None, 4, 16, losses)
    expected = .5 * (encoded.loss_terms(4)["vq_loss"] + encoded.loss_terms(16)["vq_loss"])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert encoded.continuous.grad[:, :16].abs().sum() > 0
    assert encoded.continuous.grad[:, 16:].eq(0).all()
    assert row["vq_loss"] == float(expected.detach())
    assert state.frame_index == 0


def test_independent_budget_generator_survives_resume(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.tests.test_training import settings
    from adaptok_hr.training.checkpoint import restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    plan = {**settings(), "budget_seed": 123}
    trainer = JointTrainer(small_model, plan)
    trainer.budgets(1)
    sampler = ClipSampler([{"original_frames": 64}], 3)
    config = {"model": small_model.config.to_dict(), "training": plan}
    save_training(tmp_path / "run.pt", trainer, sampler, config, "identity", 0, 0)
    expected = [trainer.budgets(i) for i in range(2, 10)]
    torch.randn(5000)
    resumed = JointTrainer(copy.deepcopy(small_model), plan)
    restore_training(tmp_path / "run.pt", resumed, sampler, config, "identity")
    actual = []
    for i in range(2, 10):
        torch.randn(5000)
        actual.append(resumed.budgets(i))
    assert actual == expected


@pytest.mark.parametrize("mode", ("shared", "per_level"))
def test_new_modes_preserve_real_prefix_bytes_and_reject_other_model(small_config, mode):
    from adaptok_hr.codec.bitstream import parse, truncate_packet
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.models.model import AdapTokHR

    model = AdapTokHR(replace(small_config, codebook_mode=mode, loss_scope="active_prefix")).eval()
    sender, receiver = Codec(model), Codec(copy.deepcopy(model))
    frame = torch.rand(1, 3, 128, 256)
    packet, state = sender.encode_frame(frame, budget=32)
    for k in small_config.budgets:
        short = truncate_packet(packet, k)
        assert parse(short).ids == parse(packet).ids[:k]
        actual, _ = receiver.decode_frame(short)
        expected, _ = model.decode_ids(torch.tensor([parse(short).ids]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    decoded, _ = receiver.decode_frame(packet)
    torch.testing.assert_close(state.rgb, decoded, atol=0, rtol=0)
    other = Codec(AdapTokHR(small_config).eval())
    with pytest.raises(ValueError, match="identity"):
        other.decode_frame(packet)


def test_per_level_quantization_preserves_lookup_and_both_gradient_paths(small_config):
    from adaptok_hr.models.vq import VectorQuantizer

    counts = (2, 8, 32)
    vq = VectorQuantizer(codebook_mode="per_level", loss_scope="active_prefix", counts=counts)
    values = torch.randn(1, 42, 16, requires_grad=True)
    encoded = vq(values)
    torch.testing.assert_close(encoded.quantized, vq.lookup(encoded.ids), atol=0, rtol=0)
    (encoded.loss_terms(32)["vq_loss"] + encoded.quantized[:, :32].square().sum()).backward()
    assert values.grad[:, :32].abs().sum() > 0 and values.grad[:, 32:].eq(0).all()
    assert all(book.weight.grad.abs().sum() > 0 for book in vq.books)


@pytest.mark.parametrize("mode", ("shared", "per_level"))
def test_training_frame_decoder_never_observes_unsent_candidates(small_config, mode):
    from adaptok_hr.models.model import AdapTokHR

    model = AdapTokHR(replace(small_config, codebook_mode=mode)).eval()
    frame = torch.rand(1, 3, 128, 256)
    encoded = model.encode_features(frame)
    for budget in model.config.budgets:
        expected, _ = model.decode_ids(encoded.ids[:, :budget])
        changed = encoded.quantized.detach().clone()
        changed[:, budget:] = float("nan")
        output = model.forward_frame(frame, None, budget, replace(encoded, quantized=changed))
        torch.testing.assert_close(output.rgb, expected, atol=0, rtol=0)


def test_checkpoint_rejects_wrong_codebook_or_loss_configuration(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.tests.test_training import settings
    from adaptok_hr.training.checkpoint import restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    trainer = JointTrainer(small_model, settings())
    sampler = ClipSampler([{"original_frames": 64}], 3)
    config = {"model": small_model.config.to_dict(), "training": settings()}
    save_training(tmp_path / "run.pt", trainer, sampler, config, "identity", 0, 0)
    for changes in ({"codebook_mode": "per_level"}, {"loss_scope": "active_prefix"}):
        model = AdapTokHR(replace(small_model.config, **changes))
        config["model"] = model.config.to_dict()
        with pytest.raises(ValueError, match="identity"):
            restore_training(tmp_path / "run.pt", JointTrainer(model, settings()), sampler, config, "identity")


def test_inference_loader_rejects_tampered_configuration_identity(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.tests.test_training import settings
    from adaptok_hr.training.checkpoint import load_model, save_training
    from adaptok_hr.training.train import JointTrainer

    config = {"model": small_model.config.to_dict(), "training": settings()}
    path = tmp_path / "run.pt"
    save_training(path, JointTrainer(small_model, settings()), ClipSampler([{"original_frames": 64}], 3),
                  config, "identity", 0, 0)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["config"]["model"]["loss_scope"] = "active_prefix"
    torch.save(state, path)
    with pytest.raises(ValueError, match="configuration identity"):
        load_model(path, device="cpu")


@pytest.mark.gpu
@pytest.mark.slow
def test_native_per_level_active_vq_and_independent_budget_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.data.dataset import ClipSampler, read_clip
    from adaptok_hr.data.manifest import load_index
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.runtime import read_config, seed_everything
    from adaptok_hr.training.checkpoint import restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    seed_everything(12)
    config = read_config("experiments/adaptok_hr/fine_recovery/configs/A3.yaml")
    config["training"]["losses"]["perceptual"] = 0
    source = load_index(config["data"]["index"])["splits"]["train"][0]
    clip, _ = read_clip(source, 0, 4)
    model = AdapTokHR(ModelConfig(**config["model"])).cuda()
    trainer = JointTrainer(model, config["training"])
    sampler = ClipSampler([source], 30)
    trainer.update(clip, 1)
    save_training(tmp_path / "per_level.pt", trainer, sampler, config, "native", 1, 4)
    expected = trainer.update(clip, 2)
    weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    restored = JointTrainer(AdapTokHR(model.config).cuda(), config["training"])
    restore_training(tmp_path / "per_level.pt", restored, sampler, config, "native")
    actual = restored.update(clip, 2)
    assert actual["budgets"] == expected["budgets"]
    for name, value in weights.items():
        torch.testing.assert_close(value, restored.model.state_dict()[name].cpu(), atol=0, rtol=0)
