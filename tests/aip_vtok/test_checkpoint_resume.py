import copy

import pytest
import torch
from torch import nn

from aip_vtok.runtime import save_checkpoint
from aip_vtok.training.smoke import checkpoint_artifact, restore, trainable_state
from aip_vtok.training.trainer import RNGStreams


class SmallCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(6, 6).requires_grad_(False)
        self.projector = nn.Linear(6, 1).requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.predictor(self.backbone(x)) + self.projector(x)


def update(model, optimizer, rng):
    x = torch.randn(8, 6, generator=rng.data_rng)
    x = x * (1 + torch.rand((), generator=rng.aug_rng))
    scale = torch.rand((), generator=rng.budget_rng) + 0.5
    x[:, int(torch.randint(6, (), generator=rng.crop_rng))] *= 0.5
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - torch.sin(x.sum(1, keepdim=True)) * scale).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@pytest.fixture
def cpu_cuda_boundary(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)


def test_real_adam_resume_and_all_four_rng_streams_match_uninterrupted_training(tmp_path, cpu_cuda_boundary):
    torch.manual_seed(13)
    model = SmallCodec()
    fresh = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=0.001)
    rng = RNGStreams(1)
    update(model, optimizer, rng)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, checkpoint_artifact(1, model, optimizer, rng, [], {"seed": 1}, {"data": "abc"}))
    expected = update(model, optimizer, rng)
    resumed_optimizer = torch.optim.AdamW(fresh.predictor.parameters(), lr=0.001)
    resumed_rng = RNGStreams(999)
    restored = torch.load(path, weights_only=False)
    step, _ = restore(restored, fresh, resumed_optimizer, resumed_rng, {"seed": 1}, {"data": "abc"})
    assert step == 1
    actual = update(fresh, resumed_optimizer, resumed_rng)
    assert actual == expected
    for name, value in trainable_state(model).items():
        assert torch.equal(value, trainable_state(fresh)[name])


def test_resume_rejects_frozen_weight_injection_before_mutating_model(cpu_cuda_boundary):
    model = SmallCodec()
    optimizer = torch.optim.AdamW(model.predictor.parameters())
    rng = RNGStreams(1)
    artifact = checkpoint_artifact(0, model, optimizer, rng, [], {}, {})
    artifact["model"]["backbone.weight"] = torch.zeros_like(model.backbone.weight)
    before = model.backbone.weight.detach().clone()
    with pytest.raises(ValueError, match="trainable"):
        restore(artifact, model, optimizer, rng, {}, {})
    assert torch.equal(model.backbone.weight, before)
