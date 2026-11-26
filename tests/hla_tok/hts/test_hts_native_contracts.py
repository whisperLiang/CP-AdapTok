import os
from pathlib import Path

import pytest
import torch

from hla_tok.hts.native import NativeNetworks
from scripts.hla_tok.hts.audit_native import Point
from scripts.hla_tok.hts.native_contracts import compare_case, official_entry
from scripts.hla_tok.uf.inputs import inputs
from third_party.DCVC.src.utils.common import set_torch_env


@pytest.mark.gpu
@torch.inference_mode()
def test_cuda_rgb_and_payload_match_official_script(tmp_path):
    set_torch_env()
    reader, plans = inputs()
    clip = plans["calibration"][0]
    frames = [reader.read(clip.split, clip.video, clip.start + i) for i in range(25)]
    rgb = torch.stack(frames).cuda().float().div_(255)
    with torch.cuda.stream(torch.cuda.Stream()):
        networks = NativeNetworks(Path(os.environ["HTS_WEIGHTS"]), "hts", 0.15)
        point = Point("diagnostic", 7, 19)
        direct = official_entry()
        cases = [compare_case(direct, networks, rgb[:9], point, tmp_path / "regular"),
                 compare_case(direct, networks, rgb, point, tmp_path / "after-reset", reset_interval=16)]
    assert [fields["frames"] for fields in cases] == [9, 25]
    for fields in cases:
        assert fields["payload_and_nal_parity"]
        assert fields["rgb_parity"]
        assert fields["context_parity"]
        assert fields["reference_parity"]
    assert reader.counts == {"train": 25, "validation": 0, "test": 0}
