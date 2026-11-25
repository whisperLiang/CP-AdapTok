from dataclasses import replace

import pytest

from hla_tok.uf.base import FrameInfo, ReferenceOrder, schedule


def test_configured_i_refresh_and_feature_reset_are_distinct():
    main = [schedule(index, intra_period=32, reset_interval=32) for index in range(66)]
    default = [schedule(index, intra_period=0, reset_interval=32) for index in range(66)]
    assert [index for index, (intra, _) in enumerate(main) if intra] == [0, 32, 64]
    assert [index for index, (intra, _) in enumerate(default) if intra] == [0]
    assert [index for index, (_, reset) in enumerate(default) if reset] == [32, 64]


def test_reference_gap_stays_desynchronized_until_real_i_frame():
    order = ReferenceOrder()
    intra = FrameInfo(0, 1080, 1920, 0, True, 1, False)
    order.check(intra)
    order.commit(intra)
    with pytest.raises(ValueError, match="DESYNC"):
        order.check(replace(intra, index=2, intra=False))
    with pytest.raises(ValueError, match="DESYNC"):
        order.check(replace(intra, index=3, intra=False, reset=True))
    recovery = replace(intra, index=4)
    order.check(recovery)
    order.commit(recovery)
    order.check(replace(intra, index=5, intra=False))
    with pytest.raises(ValueError, match="duplicate"):
        order.check(recovery)


@pytest.mark.parametrize("changes", [{"qp": 64}, {"parallel": 0}, {"parallel": 9}, {"height": 0}, {"reset": True}])
def test_rejects_invalid_official_frame_fields(changes):
    with pytest.raises(ValueError):
        replace(FrameInfo(0, 1080, 1920, 0, True, 1, False), **changes)
