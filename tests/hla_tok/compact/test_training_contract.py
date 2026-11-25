import pytest
import torch

from hla_tok.compression.profiles import Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactTransform
from hla_tok.streaming.compact_wire import max_budget
from scripts.hla_tok.compact.adapter import adapter_latents, budget_choices


def test_rates_use_physical_header_and_do_not_overweight_saturation():
    profile = Profile.Q4_4
    caps = [20833, 26520, 28080, 41666, 83333, 166666]
    result = budget_choices(profile, caps)
    assert result == (10386, 13230, 14010, 16320)
    assert all(max_budget(cap, profile) in result for cap in caps)


def test_hard_training_gradients_reach_only_lift_and_input_is_explicit_integer():
    t = CompactTransform.from_pca(torch.zeros(1, 32), torch.eye(32)[None, :, :4])
    t.requires_grad_(True)
    scalar = StaticScalar(torch.full((1, 4), 0.25), 4)
    values = torch.full((1, 3, 32), 0.24)
    ranks = torch.arange(3)
    noisy = adapter_latents(t, scalar, values, ranks, soft=True)
    assert noisy.requires_grad and not torch.equal(noisy, values[..., :4])
    hard = adapter_latents(t, scalar, values, ranks, soft=False)
    assert torch.equal(hard, torch.full((1, 3, 4), 0.25))
    assert not hard.requires_grad
    t.decode(hard, ranks).sum().backward()
    assert t.projection.weight.grad is None
    assert t.lift.weight.grad is not None
    assert all(x.grad is None for x in scalar.buffers())


def test_a_different_asset_cannot_resume_same_training_identity(tmp_path):
    from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    path = tmp_path / "step.pt"
    identity = ResumeIdentity("T1", "profile-a", "train-lock", "base")
    save_checkpoint(path, identity, model, opt, scheduler, torch.Generator().manual_seed(7), step=1, data_cursor=1, counters={})
    assert read_checkpoint(path, identity).state["step"] == 1
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, ResumeIdentity("T1", "profile-b", "train-lock", "base"))


def test_frozen_c0_attestation_detects_updates():
    from scripts.hla_tok.compact.adapter import assert_frozen
    from scripts.hla_tok.continuous.inputs import model_digest

    model = torch.nn.Linear(2, 2).requires_grad_(False)
    digest = model_digest(model)
    assert assert_frozen(model, digest) == digest
    with torch.no_grad():
        model.weight.add_(.25)
    with pytest.raises(ValueError, match="state_dict"):
        assert_frozen(model, digest)
    model.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        assert_frozen(model, model_digest(model))
