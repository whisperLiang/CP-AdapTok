"""Full 32D HLT scan plans with a base boundary independent of display budgets."""

import numpy as np

from hla_tok.streaming.continuous import RankOrder
from hla_tok.streaming.temporal_wire import HEADER_BYTES, RECORD_BYTES
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE


def base_scan(order: RankOrder, cap: int, rank_limit: int, components: int, bits: int,
              block_size: int) -> np.ndarray:
    ranks = []
    locations = 0
    for rank in order.ranks[:rank_limit]:
        native = LAYOUT.tokens_for_rank(rank)
        updated = locations + len(native) * components
        blocks = (updated + block_size - 1) // block_size
        if HEADER_BYTES + blocks * RECORD_BYTES + (updated * bits + 7) // 8 > cap:
            break
        ranks.append(native)
        locations = updated
    if not ranks:
        raise ValueError("base byte cap cannot cover one full native Rank")
    return (np.concatenate(ranks)[:, None] * LAYOUT.components + np.arange(components)).flatten()


def scan_table(order: RankOrder, base: np.ndarray, score: np.ndarray | None = None) -> np.ndarray:
    dim = LAYOUT.components
    if score is None:
        tokens = LAYOUT.wire_indices(order)
        fixed = tokens[:, None] * dim + np.arange(dim)
        candidates = np.concatenate((fixed[:, :16].flatten(), fixed[:, 16:].flatten()))
    else:
        if score.shape != RANK_COMPONENT_SHAPE or not np.isfinite(score).all():
            raise ValueError("scan scores must cover every native Rank and component")
        order_index = np.asarray(order.ranks)
        priority = np.argsort(-score[order_index].flatten(), kind="stable")
        ranks, components = np.divmod(priority, dim)
        candidates = np.concatenate([LAYOUT.tokens_for_rank(int(order_index[r])) * dim + c
                                     for r, c in zip(ranks, components, strict=True)])
    extra = np.ones(LAYOUT.token_count * dim, dtype=bool)
    extra[base] = False
    return np.concatenate((base, candidates[extra[candidates]])).astype(np.int32)
