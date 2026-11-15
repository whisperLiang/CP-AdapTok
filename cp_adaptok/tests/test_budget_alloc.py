import pytest

from cp_adaptok.streaming.online_allocator import OnlineTokenAllocator, RDPoint, offline_oracle


def test_allocator_respects_real_bytes_deadline_and_base():
    points = [RDPoint(128, 300, .1), RDPoint(256, 550, .05), RDPoint(512, 1100, .03)]
    a = OnlineTokenAllocator(k_ref=128)
    assert a.allocate(points, bandwidth_bps=8000, deadline_s=.6).tokens == 256
    assert a.allocate(points, bandwidth_bps=8000, deadline_s=.2) is None
    assert a.allocate(points, bandwidth_bps=8000, deadline_s=.6, queued_bytes=500) is None
    assert a.allocate(points, bandwidth_bps=8000, deadline_s=2, mode="fixed", fixed_k=128).tokens == 128
    with pytest.raises(ValueError):
        a.allocate(points, bandwidth_bps=-1, deadline_s=1)


def test_oracle_is_global_and_offline():
    frames = [[RDPoint(128, 100, .1), RDPoint(256, 200, .01)],
              [RDPoint(128, 100, .2), RDPoint(256, 200, .19)]]
    assert [p.tokens for p in offline_oracle(frames, total_bytes=300)] == [256, 128]
