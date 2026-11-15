"""Actual deterministic tokens, causal validation and train-only calibration on tiny real decodes."""

import copy
import json

import cv2
import numpy as np
import pytest
import torch

from cp_adaptok.large_scale_data import (
    audit_source,
    container_metadata,
    load_manifests,
    validate_training_source,
)
from cp_adaptok.large_scale_validation import calibrate_entropy, validate_checkpoint


def sources(tmp_path):
    rows = []
    for index in range(3):
        folder = tmp_path / str(index)
        folder.mkdir()
        for t in range(6):
            pixels = np.random.default_rng(index * 100 + t).integers(0, 255, (16, 24, 3), dtype=np.uint8)
            cv2.imwrite(str(folder / f"{t:05d}.png"), pixels)
        rows.append(audit_source(folder, f"train-source:{index}", "official", "train"))
    return rows


def configuration(rows):
    from cp_adaptok.large_scale_data import video_specs

    splits = {"train": rows[:2], "validation": [dict(rows[2], role="validation")], "test": []}
    return {"seed": 47, "data": {k: video_specs(v) for k, v in splits.items()},
            "large_scale": {"entropy_calibration": {"clips": 2, "frames": 4, "crop": [16, 20], "pseudocount": 1}},
            "evaluation": {"budgets": [8, 16, 24, 32], "intra_period": 32, "save_frames": False,
                           "metrics": ["psnr"]}}, splits


def test_train_only_calibration_repeats_and_preserves_representation(codec, tmp_path):
    config, _ = configuration(sources(tmp_path))
    model = copy.deepcopy(codec)
    queries = model.tokenizer.queries.detach().clone()
    first = calibrate_entropy(model, config)
    second = calibrate_entropy(model, config)
    assert first["entropy_id"] == second["entropy_id"]
    assert second["observed_tokens"] == 2 * 4 * codec.config.n_max
    assert not first["validation_or_test_tokens_used"]
    assert all(window["path"] in [r["path"] for r in config["data"]["train"]] for window in first["windows"])
    torch.testing.assert_close(queries, model.tokenizer.queries, rtol=0, atol=0)


@pytest.mark.parametrize("tier", ["quick", "full"])
def test_validation_restores_training_state_and_reports_actual_rate_only_for_rans(codec, tmp_path, tier):
    config, splits = configuration(sources(tmp_path))
    model = copy.deepcopy(codec).train()
    before = model.entropy_model.logits.detach().clone()
    result = validate_checkpoint(model, config, splits, tier, tmp_path / "validation",
                                 settings_override={"sequences": 1, "frames": 4, "crop": None, "metrics": ["psnr"]})
    assert model.training
    torch.testing.assert_close(before, model.entropy_model.logits, rtol=0, atol=0)
    assert len(result["records"]) == 4
    assert [r["frame_type"] for r in result["records"]] == ["I", "P", "P", "P"]
    assert all(row["mse"] > 0 for row in result["budget_summary"])
    assert result["real_entropy_bitstream"] == (tier == "full")
    if tier == "full":
        assert all(row["bpp"] > 0 for row in result["budget_summary"])
        assert result["cache_equivalence_verified"]
        assert result["entropy_id"] == result["entropy_calibration"]["entropy_id"]
    else:
        assert all(row["bpp"] is None for row in result["budget_summary"])


def test_container_audit_counts_the_entire_video(tmp_path):
    path = tmp_path / "native.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (24, 16))
    for index in range(21):
        writer.write(np.full((16, 24, 3), index, np.uint8))
    writer.release()
    row = container_metadata(path)
    assert row["original_frames"] == 21
    assert row["resolution"] == [24, 16]
    assert row["container_fps"] == 24
    with pytest.raises(ValueError, match="cannot open"):
        container_metadata(tmp_path / "missing.mp4")


def test_load_manifests_rejects_train_val_aliases_and_old_test_names(tmp_path):
    rows = sources(tmp_path)
    directory = tmp_path / "manifests"
    directory.mkdir()
    for split, items in {"train": rows[:2], "validation": [rows[2]], "test": []}.items():
        (directory / f"cpat_{split}.json").write_text(json.dumps({"records": items, "split": split}))
    splits, hashes = load_manifests(directory)
    assert len(splits["train"]) == 2 and len(hashes["train"]) == 64
    forbidden = dict(rows[0], content_id="UVG:beauty")
    with pytest.raises(ValueError, match="historical test"):
        validate_training_source(forbidden)
    (directory / "cpat_validation.json").write_text(json.dumps({"records": [rows[0]], "split": "validation"}))
    with pytest.raises(ValueError, match="leakage"):
        load_manifests(directory)
