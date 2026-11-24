import hashlib

import cv2
import numpy as np
import pytest

from hla_tok.evaluation.data import AuditedFrames


def setup_rows(tmp_path, monkeypatch):
    # The sealed path is real, while no decoded value or file read is allowed from it.
    _, encoded = cv2.imencode(".png", np.zeros((128, 128, 3), dtype=np.uint8))
    data = encoded.tobytes()
    roles = {"train": "train", "validation": "historical-development", "test": "test"}
    by_split = {}
    for split, role in roles.items():
        root = tmp_path / split
        root.mkdir()
        (root / "000.png").write_bytes(data + split.encode())
        item = {"file": "000.png", "sha256": hashlib.sha256(data + split.encode()).hexdigest()}
        by_split[split] = [{"content_id": split, "path": str(root), "role": role,
                            "resolution": [128, 128], "frame_hashes": [item]}]
    monkeypatch.setattr("hla_tok.evaluation.data.load_manifests",
                        lambda _: (by_split, dict.fromkeys(roles, "sealed")))
    audit = AuditedFrames(tmp_path, dict.fromkeys(roles, "sealed"))
    return audit, by_split


def test_only_allowed_splits_decode_and_counts_track_actual_reads(tmp_path, monkeypatch):
    audit, _ = setup_rows(tmp_path, monkeypatch)
    valid = audit.read("validation", "validation", 0)
    assert valid.shape == (3, 128, 128)
    with pytest.raises(PermissionError):
        audit.read("test", "test", 0)
    with pytest.raises(KeyError):
        audit.read("train", "test", 0)
    assert audit.counts == {"train": 0, "validation": 1, "test": 0}


def test_sealed_test_alias_is_rejected_before_decode(tmp_path, monkeypatch):
    audit, sources = setup_rows(tmp_path, monkeypatch)
    alias = tmp_path / "validation" / "000.png"
    alias.unlink()
    alias.symlink_to(tmp_path / "test" / "000.png")
    sources["validation"][0]["frame_hashes"][0]["sha256"] = sources["test"][0]["frame_hashes"][0]["sha256"]
    with pytest.raises(PermissionError):
        audit.read("validation", "validation", 0)
    assert audit.counts == {"train": 0, "validation": 0, "test": 0}
