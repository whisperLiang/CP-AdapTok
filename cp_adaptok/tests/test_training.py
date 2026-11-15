import copy

import pytest
import torch

from cp_adaptok import training
from cp_adaptok.runtime import load_checkpoint, save_checkpoint
from cp_adaptok.training import configure_stage, train_clip


def test_short_bptt_and_checkpoint_resume_match_an_uninterrupted_update(codec, tmp_path):
    config = {
        "stage": 2,
        "training": {"tbptt": 2, "amp_bf16": False},
        "loss": {"mse": 1.0, "vq": 0.1, "latent": 0.0, "estimated_rate": 0.0001, "monotonic": 0.1},
    }
    model = copy.deepcopy(codec).train()
    configure_stage(model, 2, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    torch.manual_seed(100)
    clip = torch.rand(4, 3, 16, 24)

    def update(network, opt):
        opt.zero_grad(set_to_none=True)
        records = train_clip(network, clip, [8, 32], config, {}, None, 1.0)
        opt.step()
        return records

    initial = model.tokenizer.queries.detach().clone()
    update(model, optimizer)
    assert not torch.equal(initial, model.tokenizer.queries)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, torch.Generator(), 1, config, "smoke")
    uninterrupted = update(model, optimizer)
    resumed = copy.deepcopy(codec).train()
    configure_stage(resumed, 2, config)
    resumed_opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=0.001)
    artifact = load_checkpoint(checkpoint)
    resumed.load_state_dict(artifact["model"])
    resumed_opt.load_state_dict(artifact["optimizer"])
    continued = update(resumed, resumed_opt)
    torch.testing.assert_close(resumed.tokenizer.queries, model.tokenizer.queries, rtol=0, atol=0)
    assert continued == uninterrupted


def test_resume_restores_rng_after_auxiliary_network_construction(codec, tmp_path, monkeypatch):
    model = copy.deepcopy(codec)
    config = {
        "seed": 47,
        "stage": 2,
        "data": {},
        "training": {"lr": 0.001},
        "paths": {"output": str(tmp_path / "results"), "save_checkpoint": str(tmp_path / "latest.pt")},
    }
    configure_stage(model, 2, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    generator = torch.Generator().manual_seed(47)
    path = tmp_path / "completed.pt"
    save_checkpoint(path, model, optimizer, generator, 1, config, "test")
    artifact = load_checkpoint(path)

    def construct_auxiliary(*args):
        torch.rand(11)  # Teacher/perceptual constructors initialize weights before loading them.
        return None

    monkeypatch.setattr(training, "perceptual_networks", construct_auxiliary)
    monkeypatch.setattr(training, "make_teacher", construct_auxiliary)
    torch.rand(19)
    report = training.run_training(model, config, 1, artifact=artifact, resume=True, status="test")
    assert report["completed_steps"] == 1
    assert torch.equal(torch.get_rng_state(), artifact["torch_rng"])


def test_stage_three_updates_entropy_and_scorer_with_tokenizer_frozen(codec):
    model = copy.deepcopy(codec)
    configure_stage(model, 3, {})
    queries = model.tokenizer.queries.detach().clone()
    logits = model.entropy_model.logits.detach().clone()
    features = torch.randn(1, model.config.n_max, model.config.dim)
    before = model.scorer(features, [8, 32]).detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = torch.zeros(1, model.config.n_max, dtype=torch.long)
    loss = model.entropy_model.loss(ids, model.config.packet_tokens)
    loss = loss + model.scorer(features, [8, 32]).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(model.entropy_model.logits, logits)
    assert not torch.equal(model.scorer(features, [8, 32]), before)
    assert torch.equal(model.tokenizer.queries, queries)
    assert all(p.grad is None for p in model.ld.parameters())


def test_stage_four_only_unfreezes_requested_backbone_modules(codec):
    model = copy.deepcopy(codec)
    configure_stage(model, 4, {"training": {"fine_tune": ["ld.model.encoder"]}})
    assert all(p.requires_grad for p in model.ld.model.encoder.parameters())
    assert not any(p.requires_grad for p in model.ld.model.feature_extractor.parameters())
    assert not any(p.requires_grad for p in model.intra.parameters())
    assert all(p.requires_grad for p in model.vq.parameters())
    with pytest.raises(ValueError, match="stages 1-4"):
        configure_stage(model, 0, {})
