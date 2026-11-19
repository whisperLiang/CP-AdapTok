import hashlib

import numpy as np
import pytest
from PIL import Image

from aip_vtok.training.data import read_window, sources


def source(tmp_path):
    frames = []
    for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
        path = tmp_path / f"{index:05d}.png"
        Image.new("RGB", (16, 8), color).save(path)
        frames.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"path": str(tmp_path), "resolution": [16, 8], "original_frames": 2, "frame_hashes": frames}


def test_reads_manifest_names_even_if_unlisted_image_is_inserted_in_directory(tmp_path):
    row = source(tmp_path)
    Image.new("RGB", (16, 8), (0, 255, 0)).save(tmp_path / "00000_extra.png")
    frame = read_window(row, 1, 1)[0]
    assert np.array_equal(frame[:, 0, 0].numpy(), [0, 0, 255])


def test_rejects_unhashed_frames_wrong_dimensions_and_source_mutation(tmp_path):
    row = source(tmp_path)
    row["frame_hashes"] = row["frame_hashes"][:1]
    with pytest.raises(ValueError, match="manifest"):
        read_window(row, 0, 2)
    row = source(tmp_path)
    row["resolution"] = [1920, 1080]
    with pytest.raises(ValueError, match="resolution"):
        read_window(row, 0, 1)
    row = source(tmp_path)
    Image.new("RGB", (16, 8), (0, 255, 0)).save(tmp_path / "00000.png")
    with pytest.raises(ValueError, match="manifest"):
        read_window(row, 0, 1)


def test_sealed_test_access_fails_before_any_path_is_opened():
    with pytest.raises(ValueError, match="sealed"):
        sources({"manifest_dir": "/missing-directory"}, "test")
