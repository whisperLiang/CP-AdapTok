import pytest
import torch

from hla_tok.hts.native import frame_input
from hla_tok.hts.units import NativeSchedule
from scripts.hla_tok.uf.inputs import inputs


def test_native_input_reads_only_the_coded_unit_and_copies_the_tail():
    source = torch.linspace(0, 1, 5).view(5, 1, 1, 1).expand(-1, 3, 16, 32)
    intra, tail = NativeSchedule(8).units(len(source))
    i_original = frame_input(source, intra)
    assert torch.equal(i_original, frame_input(source[:1], intra))
    tail_frames = frame_input(source, tail).split(3, dim=1)
    assert len(tail_frames) == 8
    assert not torch.equal(tail_frames[2], tail_frames[3])
    assert all(torch.equal(tail_frames[3], repeated) for repeated in tail_frames[4:])


def test_frozen_metadata_preserves_cohorts_and_seals_test_pixels():
    reader, splits = inputs()
    assert sum(clip.length for clip in splits["validation"]) == 174
    assert sum(clip.length for clip in splits["calibration"]) == 128
    assert {c.video for c in splits["fit"]}.isdisjoint(c.video for c in splits["calibration"])
    with pytest.raises(PermissionError, match="Test remains sealed"):
        reader.read("test", "any", 0)
    assert reader.counts == {"train": 0, "validation": 0, "test": 0}
