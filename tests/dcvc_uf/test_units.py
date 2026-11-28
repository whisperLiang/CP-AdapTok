import pytest

from baselines.dcvc_uf.units import NativeSchedule, Unit, pad_indices


def test_first_frame_stands_alone_and_full_units_keep_order():
    units = list(NativeSchedule(8, reset_interval=128).units(174))
    assert units[:2] == [Unit(0, 1, 1, True, False), Unit(1, 8, 8, False, False)]
    assert [index for unit in units for index in unit.indices] == list(range(174))
    assert [unit.start for unit in units if unit.reset] == [121]
    assert units[-1] == Unit(169, 5, 8, False, False)


@pytest.mark.parametrize('tail', range(1, 8))
def test_tail_repeats_last_index_and_preserves_valid_count(tail):
    intra, p = NativeSchedule(8).units(1 + tail)
    assert pad_indices(intra) == (0,)
    assert pad_indices(p) == (*range(1, 1 + tail), *((tail,) * (8 - tail)))
    assert p.valid == tail


def test_matched_gop_keeps_four_whole_chunks_and_explicit_i_restarts():
    units = list(NativeSchedule(8, gop_frames=33, reset_interval=32).units(74))
    assert [unit.start for unit in units if unit.intra] == [0, 33, 66]
    assert [unit.start for unit in units if unit.reset] == [25, 58]
    assert [(unit.start, unit.valid) for unit in units if unit.valid < unit.coded] == [(67, 7)]


def test_legacy_ld_schedule_and_invalid_ht_gop():
    units = list(NativeSchedule(1, gop_frames=32, reset_interval=32).units(67))
    assert [unit.start for unit in units if unit.intra] == [0, 32, 64]
    assert not any(unit.reset for unit in units)
    with pytest.raises(ValueError, match='whole P units'):
        NativeSchedule(8, gop_frames=32)


@pytest.mark.parametrize('start,valid,coded,intra,reset', [
    (-1, 1, 8, False, False),
    (1, 0, 8, False, False),
    (1, 9, 8, False, False),
    (1, 1, 8, True, False),
    (1, 1, 1, True, True),
])
def test_invalid_units_are_rejected(start, valid, coded, intra, reset):
    with pytest.raises(ValueError):
        Unit(start, valid, coded, intra, reset)
