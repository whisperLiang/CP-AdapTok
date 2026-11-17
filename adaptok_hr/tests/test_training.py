import copy

import pytest
import torch


def settings():
    return {"learning_rate": 1e-4, "weight_decay": .01, "gradient_clip": 1.0, "amp_bf16": False,
            "tbptt": 2, "nested_every": 1, "budget_mode": "nested", "fixed_budget": 32,
            "losses": {"l1": 1., "mse": 1., "perceptual": 0., "vq": 1., "temporal": .1, "monotonic": .1}}


def test_optimizer_resume_matches_uninterrupted_and_sampling(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.training.checkpoint import restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    records = [{"original_frames": 64}, {"original_frames": 128}]
    sampler = ClipSampler(records, 42)
    config = {"model": small_model.config.to_dict(), "training": settings()}
    trainer = JointTrainer(small_model, config["training"])
    clip = torch.rand(2, 3, 128, 256)
    trainer.update(clip, 1)
    sampler.plan(2)
    save_training(tmp_path / "one.pt", trainer, sampler, config, "manifest-hash", 1, 2)
    expected_plan = sampler.plan(2)
    expected_row = trainer.update(clip, 2)
    expected = copy.deepcopy(trainer.model.state_dict())
    resumed = JointTrainer(copy.deepcopy(small_model), config["training"])
    sampling = ClipSampler(records, 99)
    meta = restore_training(tmp_path / "one.pt", resumed, sampling, config, "manifest-hash")
    assert meta["completed_steps"] == 1 and meta["training_frames"] == 2
    assert sampling.plan(2) == expected_plan
    actual_row = resumed.update(clip, 2)
    for key, value in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, atol=0, rtol=0)
    assert actual_row["budgets"] == expected_row["budgets"]
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    with pytest.raises(ValueError, match="manifest|identity"):
        restore_training(tmp_path / "one.pt", resumed, sampling, config, "wrong-manifest")


def test_log_rollback_discards_only_uncommitted_steps(tmp_path):
    from adaptok_hr.training.checkpoint import reconcile_log

    log = tmp_path / "training.jsonl"
    log.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":')
    assert reconcile_log(log, 2) == 1
    assert log.read_text() == '{"step": 1}\n{"step": 2}\n'
    with pytest.raises(ValueError):
        reconcile_log(log, 4)


def test_auxiliary_budget_does_not_change_committed_reference(small_model):
    from adaptok_hr.training.losses import ReconstructionLoss
    from adaptok_hr.training.train import frame_objective

    model = small_model.train()
    frame = torch.rand(1, 3, 128, 256)
    losses = ReconstructionLoss(settings()["losses"], "cpu")
    _, actual, row = frame_objective(model, frame, None, None, 4, 32, losses)
    expected = model.forward_frame(frame, None, 4)
    torch.testing.assert_close(actual.rgb, expected.state.rgb, atol=0, rtol=0)
    assert row["main_budget"] == 4 and row["auxiliary_budget"] == 32


def test_frozen_tokenizer_scorer_stage_changes_only_scorer(small_model):
    from adaptok_hr.models.scorer import BudgetScorer, scorer_update

    model = small_model.eval()
    before = copy.deepcopy(model.state_dict())
    scorer = BudgetScorer(16, model.config.budgets)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    weights = copy.deepcopy(scorer.state_dict())
    result = scorer_update(model, scorer, optimizer, torch.rand(1, 3, 128, 256), None)
    assert result["loss"] >= 0
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.parameters())
    assert any(not torch.equal(scorer.state_dict()[k], v) for k, v in weights.items())
    assert scorer.select(torch.zeros(1, 16), maximum_tokens=8) in (4, 8)


def test_adaptive_selection_produces_real_decodable_bytes(small_model):
    from adaptok_hr.codec.bitstream import parse
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.models.scorer import BudgetScorer

    codec = Codec(small_model.eval())
    scorer = BudgetScorer(16, small_model.config.budgets).eval()
    packet, sender = codec.encode_adaptive_frame(torch.rand(1, 3, 128, 256), None, scorer,
                                                  maximum_tokens=8, rate_weight=.001)
    decoded, _ = codec.decode_frame(packet, None)
    assert len(parse(packet).ids) in (4, 8)
    assert torch.equal(decoded, sender.rgb)
