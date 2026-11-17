import pytest
import torch

from experiments.fixed2048.data import read_clip, records


def test_manifest_selection_rejects_wrong_resolution_or_unknown_video(monkeypatch):
    import experiments.fixed2048.data as data

    exact = {"content_id": "valid", "resolution": [1920, 1080]}
    small = {"content_id": "small", "resolution": [1280, 720]}
    monkeypatch.setattr(
        data, "load_manifests", lambda path: ({"validation": [exact, small]}, {"validation": "hash"})
    )
    assert records({"manifest_dir": "unused"}, "validation", ["valid"])[0] == [exact]
    with pytest.raises(ValueError, match="missing"):
        records({"manifest_dir": "unused"}, "validation", ["small"])
    with pytest.raises(ValueError, match="missing"):
        records({"manifest_dir": "unused"}, "validation", ["unknown"])
    monkeypatch.setattr(
        data, "load_manifests", lambda path: ({"validation": [small]}, {"validation": "hash"})
    )
    with pytest.raises(ValueError, match="empty"):
        records({"manifest_dir": "unused"}, "validation")


def test_native_pixels_cannot_be_upsampled_and_hash_drift_is_rejected(tmp_path, monkeypatch):
    import experiments.fixed2048.data as data

    frame = tmp_path / "0.jpg"
    frame.write_bytes(b"actual frame")
    row = {
        "path": str(tmp_path),
        "content_id": "train:a",
        "original_frames": 1,
        "frame_hashes": [{"file": "0.jpg", "sha256": data.file_hash(frame)}],
    }
    monkeypatch.setattr(data, "read_video", lambda *a, **k: torch.zeros(1, 3, 8, 12))
    with pytest.raises(ValueError, match="upsampling"):
        read_clip(row, 1, crop=(1080, 1920))
    with pytest.raises(ValueError, match="actual native"):
        read_clip(row, 1)
    frame.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        read_clip(row, 1)
    with pytest.raises(ValueError, match="exceeds"):
        read_clip(row, 2)
