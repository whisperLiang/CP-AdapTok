"""Behavioral gates for fair sampling, exact resume, and stage transitions."""

import copy

import cv2
import numpy as np
import pytest
import torch

from cp_adaptok.data import check_splits
from cp_adaptok.runtime import load_checkpoint, save_checkpoint


def clip_config(tmp_path):
    folders = []
    for sequence in range(3):
        folder = tmp_path / f"video{sequence}"
        folder.mkdir()
        for frame in range(12):
            pixels = np.random.default_rng(sequence * 100 + frame).integers(0, 256, (24, 32, 3), np.uint8)
            cv2.imwrite(str(folder / f"{frame:05d}.png"), pixels)
        folders.append({"path": str(folder), "frames": 12, "content_id": f"source:{sequence}"})
    return {"data": {"train": folders}, "training": {"crop": [16, 20], "sequence_length": 4,
            "gradient_accumulation": 2, "horizontal_flip": True}}


def test_budget_draws_cannot_change_data_windows_or_augmentations(tmp_path):
    from cp_adaptok.data import TrainingClipLoader, plan_training_samples
    from cp_adaptok.tokenizer.prefix_masking import sample_budgets, sample_nested_budgets
    from cp_adaptok.training_state import TrainingRNGStreams

    config = clip_config(tmp_path)
    baseline, nested = TrainingRNGStreams(73), TrainingRNGStreams(73)
    with TrainingClipLoader(cache_bytes=4096) as loader:
        starts = []
        for _ in range(10):
            sample_budgets([8, 16, 24, 32, 40], baseline.budget_rng)
            sample_nested_budgets([8, 16, 24, 32, 40], nested.budget_rng)
            a = plan_training_samples(config, 4, [16, 20], baseline)
            b = plan_training_samples(config, 4, [16, 20], nested)
            assert a == b
            starts.extend((p.spec.path, p.spec.start) for p in a)
            for left, right in zip(loader.load(a, prefetch=False), loader.load(b, prefetch=True)):
                torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
                assert left[2] == right[2]
        assert len(set(starts)) > 10
        assert loader.cached_bytes <= 4096


def test_all_sampling_streams_and_scheduler_resume_exactly(codec, tmp_path):
    from cp_adaptok.training import configure_stage, restore_training
    from cp_adaptok.training_state import TrainingRNGStreams, make_scheduler

    model = copy.deepcopy(codec).train()
    config = {"stage": 2, "training": {"scheduler": {"warmup_steps": 2, "total_steps": 8,
              "min_lr_ratio": 0.1}, "lr": 0.001}}
    configure_stage(model, 2, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    scheduler = make_scheduler(optimizer, config["training"])
    streams = TrainingRNGStreams(47)
    for _ in range(3):
        optimizer.zero_grad()
        model.tokenizer.queries.square().mean().backward()
        optimizer.step()
        scheduler.step()
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, streams, 3, config, "trained", scheduler=scheduler)
    expected = {name: torch.rand(5, generator=rng) for name, rng in streams.items()}
    artifact = load_checkpoint(path)
    resumed_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    resumed_scheduler = make_scheduler(resumed_opt, config["training"])
    resumed = TrainingRNGStreams(999)
    assert restore_training(model, resumed_opt, resumed, artifact, 2, resumed_scheduler) == 3
    assert resumed_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for name, rng in resumed.items():
        torch.testing.assert_close(torch.rand(5, generator=rng), expected[name], rtol=0, atol=0)


def test_transferred_weights_are_not_rescaled_or_reinitialized(codec):
    from cp_adaptok.training import initialize_queries

    original = codec.tokenizer.queries.detach().clone()
    model = copy.deepcopy(codec)
    initialize_queries(model, {"query_initialization_scale": 50}, False, transferring=True)
    torch.testing.assert_close(original, model.tokenizer.queries, rtol=0, atol=0)


def test_hash_identity_prevents_cross_split_copies():
    with pytest.raises(ValueError, match="leakage"):
        check_splits({"train": [{"path": "train-copy", "sha256": "abc"}],
                      "validation": [{"path": "val-copy", "sha256": "abc"}]})


def test_quantization_stream_isolated_from_global_torch_rng(codec):
    from cp_adaptok.training_state import TrainingRNGStreams

    a, b = copy.deepcopy(codec.vq).train(), copy.deepcopy(codec.vq).train()
    a.set_training_policy("stochastic_training", 0.5)
    b.set_training_policy("stochastic_training", 0.5)
    a.sampling_generator = TrainingRNGStreams(49).quantization_rng
    b.sampling_generator = TrainingRNGStreams(49).quantization_rng
    features = torch.randn(1, 32, codec.config.dim)
    left = a(features).ids
    torch.rand(1000)
    assert torch.equal(left, b(features).ids)


def test_selection_rewards_absolute_quality_not_manufactured_prefix_gain():
    from cp_adaptok.large_scale_validation import selection_score

    def report(base, maximum):
        return {"budget_summary": [{"tokens": 128, "mse": base}, {"tokens": 2048, "mse": maximum}],
                "quality_drop_ratio": 0}

    assert selection_score(report(0.01, 0.009)) < selection_score(report(0.03, 0.008))


def test_complete_optimizer_resume_replays_stochastic_video_training(codec, tmp_path):
    from cp_adaptok.training import run_training

    config = clip_config(tmp_path)
    config.update(seed=37, stage=2, loss={"mse": 1.0, "latent": 0, "vq": 0.1,
                                       "monotonic": 0.1, "estimated_rate": 0}, teacher={"policy": "none"})
    config["training"].update(lr=0.001, tbptt=2, amp_bf16=False, clip_grad_norm=1,
                              budgets=[8, 16, 24, 32], save_every=1, validate_every=100,
                              diagnostic_steps=[], diagnostic_every=1000,
                              scheduler={"warmup_steps": 1, "total_steps": 3},
                              vq={"strategy": "stochastic_training", "temperature": 0.1})
    config["paths"] = {"output": str(tmp_path / "split"), "save_checkpoint": str(tmp_path / "split.pt")}
    uninterrupted, resumed = copy.deepcopy(codec), copy.deepcopy(codec)
    run_training(resumed, config, 1)
    artifact = load_checkpoint(tmp_path / "split.pt")
    resumed.load_state_dict(artifact["model"])
    run_training(resumed, config, 3, artifact, resume=True)
    config["paths"] = {"output": str(tmp_path / "whole"), "save_checkpoint": str(tmp_path / "whole.pt")}
    run_training(uninterrupted, config, 3)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, uninterrupted.state_dict()[name], rtol=0, atol=0)
    split = load_checkpoint(tmp_path / "split.pt")
    whole = load_checkpoint(tmp_path / "whole.pt")
    assert split["step"] == whole["step"] == 3
    for name in split["rng_streams"]:
        assert torch.equal(split["rng_streams"][name], whole["rng_streams"][name])


def test_native_manifest_rejects_gapped_frames_and_validation_training(tmp_path):
    from cp_adaptok.large_scale_data import audit_source, validate_records

    folder = tmp_path / "images"
    folder.mkdir()
    for t in (0, 1, 3, 4):
        cv2.imwrite(str(folder / f"{t:05d}.png"), np.zeros((16, 24, 3), np.uint8))
    with pytest.raises(ValueError, match="nonconsecutive"):
        audit_source(folder, "official:x", "official", "train")
    row = {"path": str(folder), "content_id": "DAVIS2017:x", "source": "DAVIS", "official_split": "val",
           "original_frames": 4, "role": "train", "resized": False}
    with pytest.raises(ValueError, match="official train"):
        validate_records({"train": [row]})


def test_phase_configuration_forks_same_warmup_and_fixed_training_budget(tmp_path, monkeypatch):
    from cp_adaptok import large_scale

    monkeypatch.setattr(large_scale, "source_checkpoint", lambda *args: None)
    manifest = {"seed": 1, "run_id": "test", "manifest_hashes": {"train": "same"}}
    row = {"path": str(tmp_path), "content_id": "video", "resolution": [1920, 1080],
           "original_frames": 100, "role": "train", "fps": 30}
    splits = {"train": [row], "validation": [], "test": []}
    a = large_scale.phase_config("s3", "B0", manifest, splits)
    b = large_scale.phase_config("s3", "B1", manifest, splits)
    assert a["training"] == b["training"]
    assert a["loss"] == b["loss"]
    assert a["model"] == b["model"]
    assert a["data"] == b["data"]
    assert a["experiment"]["target_updates"] == b["experiment"]["target_updates"] == 8000
    assert a["training"]["crop"] == [1080, 1920]
    assert a["training"]["vq"]["initialization"] is None


def test_resume_rejects_phase_and_input_changes():
    from cp_adaptok.large_scale import restore_contract

    identity = dict(phase="s2", branch="B0", config_hash="a", manifest_hashes={"train": "b"}, structure_hash="c")
    config = {"experiment": identity}
    artifact = {"config": config, "rng_streams": {"data_rng": 1}, "scheduler": {"step": 2}}
    restore_contract(config, artifact)
    altered = copy.deepcopy(config)
    altered["experiment"]["phase"] = "s3"
    with pytest.raises(ValueError, match="phase"):
        restore_contract(altered, artifact)
