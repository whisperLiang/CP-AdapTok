import copy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch


def row(path, split="train"):
    return {"path": str(path), "content_id": str(path), "sha256": hashlib.sha256(str(path).encode()).hexdigest(),
            "resolution": [1920, 1080], "original_frames": 64, "role": split,
            "official_split": "train" if split == "train" else "val", "resized": False,
            "source": "local-authorized", "fps": 30}


def test_manifest_rejects_low_resolution_and_cross_source_leaks(tmp_path):
    from adaptok_hr.data.manifest import validate_index

    a, b = row(tmp_path / "a"), row(tmp_path / "b", "validation")
    index = {"splits": {"train": [a], "validation": [b], "test": []}}
    validate_index(index)
    bad = copy.deepcopy(index)
    bad["splits"]["train"][0]["resolution"] = [1280, 720]
    with pytest.raises(ValueError, match="resolution|native"):
        validate_index(bad)
    bad = copy.deepcopy(index)
    bad["splits"]["validation"][0]["sha256"] = a["sha256"]
    with pytest.raises(ValueError, match="leak"):
        validate_index(bad)


def test_reader_matches_cp_rgb_without_upsampling_and_sampler_resume(tmp_path):
    from adaptok_hr.data.dataset import ClipSampler, read_clip

    source = tmp_path / "video"
    source.mkdir()
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = 20, 80, 200
    for i in range(4):
        cv2.imwrite(str(source / f"{i:05d}.png"), rgb[:, :, ::-1])
    record = row(source)
    record["original_frames"] = 4
    clip, metadata = read_clip(record, 0, 4, 1080, 1920)
    assert tuple(clip.shape) == (4, 3, 1080, 1920)
    torch.testing.assert_close(clip[0, :, 10, 10], torch.tensor([20, 80, 200]) / 255)
    assert metadata["resized"] is False
    one = ClipSampler([record], 9)
    one.plan(2)
    saved = one.state_dict()
    expected = one.plan(2)
    two = ClipSampler([record], 0)
    two.load_state_dict(saved)
    assert two.plan(2) == expected
    cv2.imwrite(str(source / "00000.png"), np.zeros((720, 1280, 3), np.uint8))
    with pytest.raises(ValueError, match="resolution|crop|native"):
        read_clip(record, 0, 1, 1080, 1920)


def test_external_sources_require_usage_conditions(tmp_path):
    from adaptok_hr.data.manifest import load_external

    path = tmp_path / "external.json"
    path.write_text(json.dumps({"records": [row(tmp_path / "video")]}))
    with pytest.raises(ValueError, match="license|usage"):
        load_external(path)
