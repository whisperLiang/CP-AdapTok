import argparse
import json

import cv2
import numpy as np
import pytest
import torch

from baselines.dcvc_uf.units import Unit
from pht_tok import PHTTok
from pht_tok.api import centered_yuv
from pht_tok.cli import input_clip, write_outputs
from pht_tok.official_adapter import require_centered_frames


@pytest.mark.parametrize(("backend", "options", "message"), [
    ("other", {}, "backend"),
    ("ld", {"layers": 4}, "layer counts"),
    ("hts", {"reference_layers": 9}, "reference_layers"),
    ("ld", {}, "require CUDA"),
])
def test_public_api_rejects_invalid_backend_settings_without_loading_weights(backend, options, message):
    with pytest.raises(ValueError, match=message):
        PHTTok(backend, "unused", "cpu", **options)


def test_rgb_input_is_normalized_on_the_native_grid():
    values = torch.arange(48, dtype=torch.uint8).view(1, 3, 4, 4)
    assert torch.equal(centered_yuv(values, 1, torch.device("cpu")),
                       centered_yuv(values.float() / 255, 1, torch.device("cpu")))
    with pytest.raises(ValueError, match="RGB input"):
        centered_yuv(values.int(), 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="RGB frames"):
        centered_yuv(values[:, :2], 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA FP16"):
        require_centered_frames(values.half(), 1)
    with pytest.raises(ValueError, match="YCbCr frames"):
        require_centered_frames(values[:0], 1)


def test_cli_reads_only_requested_frames_and_accounts_written_bytes(tmp_path):
    images = tmp_path / "frames"
    images.mkdir()
    for i in range(2):
        assert cv2.imwrite(str(images / f"{i}.png"), np.zeros((8, 10, 3), dtype=np.uint8))
    clip, fps = input_clip(images, 1)
    assert clip.shape == (1, 3, 8, 10) and fps == 30
    with pytest.raises(ValueError, match="positive"):
        input_clip(images, 0)
    args = argparse.Namespace(output=tmp_path / "out", backend="ld", mode="progressive", benchmark=True)
    write_outputs(args, fps, [(Unit(0, 1, 1, True, False), b"unit", (4,), 1, 2)])
    ledger = json.loads((args.output / "units.json").read_text())["units"]
    assert ledger[0]["bytes"] == len((args.output / ledger[0]["file"]).read_bytes()) == 4
    assert ledger[0]["encode_ms"] == 1 and ledger[0]["decode_ms"] == 2
