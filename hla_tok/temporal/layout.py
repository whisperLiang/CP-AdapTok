"""Native coordinates shared by analysis, scan plans, closed history, and synthesis."""

from collections.abc import Iterator
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from hla_tok.models.local import TOKEN_DIM
from hla_tok.streaming.continuous import MAX_K, NATIVE, NATIVE_RANKS, REGIONS, RankOrder


class NativeLayout:
    shape = (NATIVE.height, NATIVE.width)
    grid = (NATIVE.rows, NATIVE.columns)
    token_count = MAX_K
    components = TOKEN_DIM

    def __init__(self) -> None:
        self.tokens = np.arange(MAX_K, dtype=np.int32)
        self.ranks, self.regions = self.coordinates(self.tokens)
        self.mask = self._padded(np.ones(MAX_K, dtype=bool))
        self._rank_tokens = tuple(np.split(self.tokens, np.flatnonzero(np.diff(self.ranks)) + 1))
        for array in (self.tokens, self.ranks, self.regions, self.mask, *self._rank_tokens):
            array.setflags(write=False)

    @staticmethod
    def coordinates(token):
        value = np.asarray(token)
        if value.dtype.kind not in "iu" or np.any((value < 0) | (value >= MAX_K)):
            raise ValueError("token is outside the native layout")
        return np.divmod(value, REGIONS)

    @staticmethod
    def index(rank, region):
        rank, region = np.broadcast_arrays(rank, region)
        if rank.dtype.kind not in "iu" or region.dtype.kind not in "iu":
            raise ValueError("native rank and region must be integers")
        if np.any((rank < 0) | (rank >= len(NATIVE_RANKS)) | (region < 0) | (region >= REGIONS)):
            raise ValueError("rank or region is outside the native layout")
        token = rank.astype(np.int32) * REGIONS + region.astype(np.int32)
        if np.any(token >= MAX_K):
            raise ValueError("rank and region do not identify a native token")
        return token

    def tokens_for_rank(self, rank: int) -> np.ndarray:
        if not 0 <= rank < len(self._rank_tokens):
            raise ValueError("rank is outside the native layout")
        return self._rank_tokens[rank].view()

    def wire_indices(self, order: RankOrder) -> np.ndarray:
        return np.concatenate([self.tokens_for_rank(rank) for rank in order.ranks])

    @staticmethod
    def order(native: Tensor, order: RankOrder) -> Tensor:
        return order.apply(native)

    @staticmethod
    def restore(wire: Tensor, order: RankOrder) -> Tensor:
        return order.restore_full(wire)

    @staticmethod
    def _padded(support: np.ndarray) -> np.ndarray:
        result = np.zeros((len(NATIVE_RANKS), REGIONS), dtype=bool)
        result.reshape(-1)[:MAX_K] = support
        return result

    @lru_cache(maxsize=48)
    @torch.inference_mode(False)
    def _recovery_plan(self, support: bytes, device: torch.device) -> tuple[tuple[Tensor, Tensor, int], ...]:
        mask = self._padded(np.frombuffer(support, dtype=bool))
        changes: np.ndarray = np.flatnonzero(np.any(mask[:, 1:] != mask[:, :-1], axis=0)) + 1
        boundaries = (0, *changes.tolist(), REGIONS)
        groups = []
        for first, stop in zip(boundaries, boundaries[1:]):
            ranks = np.flatnonzero(mask[:, first])
            regions = np.arange(first, stop)[:, None]
            slots = torch.as_tensor(self.index(ranks, regions).flatten(), device=device, dtype=torch.long)
            groups.append((slots, torch.as_tensor(ranks, device=device, dtype=torch.long), stop - first))
        return tuple(groups)

    def recovery_groups(self, values: Tensor, support: bytes) -> Iterator[tuple[Tensor, Tensor]]:
        if values.ndim != 3 or tuple(values.shape[1:]) != (MAX_K, TOKEN_DIM) or len(support) != MAX_K:
            raise ValueError("expected native vectors and one support flag per actual token")
        for slots, ranks, regions in self._recovery_plan(support, values.device):
            vectors = values.index_select(1, slots).reshape(values.shape[0], regions, ranks.numel(), TOKEN_DIM)
            yield vectors, ranks


LAYOUT = NativeLayout()
