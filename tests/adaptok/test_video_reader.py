import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from baselines.adaptok import cli
from baselines.adaptok.artifacts import write_video
from baselines.adaptok.runner import prepare_source
from baselines.adaptok.video_reader import manifest_sources, read_input, source_metadata
from common.video import VideoSpec, read_video


@pytest.fixture
def sequence(tmp_path):
    # A real 1080p fixture with distinct channels and consecutive frames, saved losslessly.
    for index in range(2):
        bgr = np.full((1080, 1920, 3), (20 + index, 80, 200), dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / f"{index:05d}.png"), bgr)
    return tmp_path


def test_images_match_cp_rgb_pixels_and_manifest_hash(sequence):
    pixels, metadata = read_input(sequence, frames=2, start=0)
    expected = read_video(VideoSpec(str(sequence), frames=2))
    assert torch.equal(pixels, expected)
    assert pixels[0, :, 0, 0].tolist() == expected[0, :, 0, 0].tolist()
    hashes = [hashlib.sha256(p.read_bytes()).digest() for p in sorted(sequence.glob("*.png"))]
    assert metadata["sha256"] == hashlib.sha256(b"".join(hashes)).hexdigest()
    assert metadata["resize"] is False


def test_changed_input_is_rejected(sequence):
    metadata = source_metadata(sequence)
    metadata.update(content_id="test:images", sha256="0" * 64)
    with pytest.raises(ValueError, match="content changed"):
        prepare_source(metadata)


def test_lossless_rgb_mp4_roundtrip_uses_cp_reader(sequence, tmp_path):
    frames, _ = read_input(sequence, frames=2)
    destination = tmp_path / "roundtrip.mp4"
    write_video(destination, frames, 30)
    actual, metadata = read_input(destination, frames=2)
    assert torch.equal(frames, actual)
    assert metadata["fps"] == 30


def test_raw_yuv_range_is_explicit(tmp_path):
    path = tmp_path / "black.yuv"
    y = np.full((1080, 1920), 16, dtype=np.uint8)
    uv = np.full((1080 // 2, 1920), 128, dtype=np.uint8)
    path.write_bytes(y.tobytes() + uv.tobytes())
    with pytest.raises(ValueError, match="requires"):
        read_input(path, frames=1)
    kwargs = dict(width=1920, height=1080, pixel_format="yuv420p", color_range="limited", color_matrix="bt709")
    frames, metadata = read_input(path, frames=1, raw_yuv=kwargs)
    assert frames.shape == (1, 3, 1080, 1920)
    assert torch.count_nonzero(frames) == 0
    assert metadata["color_range"] == "limited"


def test_manifest_selection_is_by_content_id():
    path = Path(__file__).resolve().parents[3] / "data/manifests/cpat_test.json"
    if not path.exists():
        pytest.skip("sealed local data manifest is not distributed in Git")
    selected = manifest_sources(path, ["DAVIS2017:cows", "DAVIS2017:goat", "DAVIS2017:parkour"])
    assert [r["content_id"] for r in selected] == ["DAVIS2017:cows", "DAVIS2017:goat", "DAVIS2017:parkour"]
    with pytest.raises(ValueError, match="missing content"):
        manifest_sources(path, ["does-not-exist"])


def test_cli_image_sequence_preserves_declared_fps_and_native_size(sequence):
    args = SimpleNamespace(input=str(sequence), fps=24)
    record = cli.input_records(args, {})[0]
    assert record["fps"] == 24 and record["source_frames"] == 2
    assert (record["width"], record["height"]) == (1920, 1080)
    assert record["path"] == str(sequence.resolve())
    assert record["content_id"] == f"single:{sequence.stem}"


def test_cli_yuv_never_guesses_color_range_or_matrix(tmp_path):
    path = tmp_path / "source.yuv"
    path.write_bytes(bytes([16]) * (1920 * 1080) + bytes([128]) * (1920 * 1080 // 2))
    args = SimpleNamespace(input=str(path), fps=25, raw_width=1920, raw_height=1080,
                           raw_pixel_format="yuv420p", raw_color_range=None, raw_color_matrix=None)
    with pytest.raises(ValueError, match="requires"):
        cli.input_records(args, {})
    args.raw_color_range, args.raw_color_matrix = "limited", "bt709"
    record = cli.input_records(args, {})[0]
    assert record["fps"] == 25
    assert record["raw_yuv"] == {"width": 1920, "height": 1080, "pixel_format": "yuv420p",
                                  "color_range": "limited", "color_matrix": "bt709"}
