import numpy as np
import pytest

from hla_tok.streaming.continuous import NATIVE_RANKS, Order, RankOrder
from hla_tok.streaming.temporal_wire import HEADER_BYTES, RECORD_BYTES
from hla_tok.temporal.allocation import base_scan, scan_table
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE


@pytest.mark.parametrize("bits,block", [(4, 64), (4, 128), (4, 256), (8, 256)])
def test_base_scan_fits_raw_limit_and_scan_uses_full_component_space(bits, block):
    ranks = NATIVE_RANKS[::2] + NATIVE_RANKS[1::2]
    rank_order = RankOrder(Order.O2, ranks)
    cap = 4166
    base = base_scan(rank_order, cap, 100, 2, bits, block)
    total = len(base)
    assert HEADER_BYTES + (total + block - 1) // block * RECORD_BYTES + (total * bits + 7) // 8 <= cap
    score = np.zeros(RANK_COMPONENT_SHAPE)
    score[-1, -1] = 10
    for optional in (None, score):
        combined = scan_table(rank_order, base, optional)
        assert np.array_equal(combined[:total], base)
        assert np.array_equal(np.sort(combined), np.arange(LAYOUT.token_count * LAYOUT.components))
        if optional is not None:
            expected = LAYOUT.tokens_for_rank(ranks[-2])  # Actual final native Rank is checked independently.
            tail = LAYOUT.tokens_for_rank(LAYOUT.mask.shape[0] - 1)
            assert len(tail) == 120 and len(expected) > 0
            assert np.array_equal(combined[total:total + len(tail)], tail * LAYOUT.components + 31)


def test_bad_base_and_rank_scores_are_rejected():
    order = RankOrder.native()
    with pytest.raises(ValueError, match="base"):
        base_scan(order, 4, 100, 1, 4, 256)
    base = base_scan(order, 4166, 48, 1, 4, 256)
    for scores in (np.ones((121, 31)), np.full(RANK_COMPONENT_SHAPE, np.nan)):
        with pytest.raises(ValueError, match="scores"):
            scan_table(order, base, scores)
