"""Native C0 continuous vectors and synthesis from a legal received prefix."""

from collections.abc import Iterable, Iterator
from enum import IntEnum
from functools import lru_cache

import torch
from torch import Tensor, nn

from hla_tok import HLATok
from hla_tok.models.codec import recover_grid, recover_groups
from hla_tok.models.local import TOKEN_DIM, LocalTokenDecoder
from hla_tok.models.spatial import RGBDecoder
from hla_tok.progressive.geometry import NATIVE_SHAPE, Geometry

NATIVE = Geometry(*NATIVE_SHAPE)
REGIONS = NATIVE.num_regions
MAX_K = 16_320
FULL_RANKS, TAIL_REGIONS = divmod(MAX_K, REGIONS)
NATIVE_RANKS = tuple(range(FULL_RANKS + 1))
FLOAT_TYPES = (torch.float32, torch.float16, torch.bfloat16)


class Order(IntEnum):
    O0 = 0
    O1 = 1
    O2 = 2


def check_budget(count: int, capacity: int = MAX_K) -> None:
    if type(count) is not int or not 1 <= count <= capacity:
        raise ValueError(f"expected integer budget in [1, {capacity}]")


def check_vectors(values: Tensor, capacity: int = MAX_K) -> None:
    if values.ndim != 3 or not values.shape[0] or values.shape[2] != TOKEN_DIM:
        raise ValueError("expected continuous vectors with shape [batch, count, 32]")
    if values.dtype not in FLOAT_TYPES:
        raise ValueError("expected FP32, FP16 or BF16 continuous vectors")
    check_budget(values.shape[1], capacity)


class RankOrder:
    """A static permutation of actual Rank spans, including the partial final Rank."""

    def __init__(self, name: Order, ranks: tuple[int, ...], device: str | torch.device = "cpu"):
        if tuple(sorted(ranks)) != NATIVE_RANKS:
            raise ValueError("Rank order must include every actual Rank exactly once")
        if name == Order.O0 and ranks != NATIVE_RANKS:
            raise ValueError("O0 must use the native Rank order")
        self.name = name
        self.ranks = ranks
        spans = []
        offsets = 0
        for rank in ranks:
            width = TAIL_REGIONS if rank == FULL_RANKS else REGIONS
            spans.append((rank, offsets, width))
            offsets += width
        self._spans = tuple(spans)
        slots = [i for rank, _, width in spans for i in range(rank * REGIONS, rank * REGIONS + width)]
        self._slots = torch.tensor(slots, dtype=torch.long, device=device)
        self._inverse_slots = self._slots.argsort()

    @classmethod
    def native(cls, device: str | torch.device = "cpu") -> "RankOrder":
        return cls(Order.O0, NATIVE_RANKS, device)

    @classmethod
    def reversed(cls, device: str | torch.device = "cpu") -> "RankOrder":
        return cls(Order.O1, NATIVE_RANKS[::-1], device)

    def apply(self, full: Tensor) -> Tensor:
        check_vectors(full)
        if full.shape[1] != MAX_K:
            raise ValueError("ordering requires the complete working sequence")
        return full if self.name == Order.O0 else full.index_select(1, self._slots.to(full.device))

    def restore_full(self, full: Tensor) -> Tensor:
        check_vectors(full)
        if full.shape[1] != MAX_K:
            raise ValueError("a full inverse requires the complete working sequence")
        return full if self.name == Order.O0 else full.index_select(1, self._inverse_slots.to(full.device))

    @lru_cache(maxsize=128)
    def _recovery_plan(self, count: int, device: torch.device) -> tuple[tuple[Tensor, Tensor, int], ...]:
        spans = sorted((rank, offset, min(width, count - offset))
                       for rank, offset, width in self._spans if offset < count)
        stops = sorted({0, REGIONS, *(width for _, _, width in spans)})
        groups = []
        for low, high in zip(stops, stops[1:]):
            choices = [(rank, offset) for rank, offset, width in spans if width >= high]
            ranks = torch.tensor([rank for rank, _ in choices], dtype=torch.long, device=device)
            offsets = torch.tensor([offset for _, offset in choices], dtype=torch.long, device=device)
            select = torch.arange(low, high, device=device)[:, None] + offsets
            groups.append((select.flatten(), ranks, high - low))
        return tuple(groups)

    def recovery_groups(self, prefix: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
        """Gather received positions by raster region and ascending original Rank."""
        batch, count, dim = prefix.shape
        for positions, ranks, regions in self._recovery_plan(count, prefix.device):
            values = prefix.index_select(1, positions).reshape(batch, regions, ranks.numel(), dim)
            yield values, ranks


@torch.inference_mode()
def encode_continuous(model: HLATok, rgb: Tensor, budget: int = MAX_K) -> Tensor:
    if tuple(rgb.shape[-2:]) != NATIVE_SHAPE:
        raise ValueError("continuous C0 requires native 1080p")
    check_budget(budget)
    return model.encode_training(rgb, budget, quantize=False).continuous


def truncate(values: Tensor, budget: int) -> Tensor:
    check_vectors(values)
    check_budget(budget, values.shape[1])
    return values[:, :budget]


class ContinuousReceiver(nn.Module):
    """Synthesis-only receiver. No source frame, encoder, or temporal state."""

    def __init__(self, local_decoder: LocalTokenDecoder, rgb_decoder: RGBDecoder):
        super().__init__()
        self.local_decoder = local_decoder
        self.rgb_decoder = rgb_decoder

    @classmethod
    def from_model(cls, model: HLATok) -> "ContinuousReceiver":
        return cls(model.local_decoder, model.rgb_decoder).eval()

    @torch.inference_mode()
    def recover(self, values: Tensor, order: RankOrder) -> Tensor:
        check_vectors(values)
        if values.shape[1] == MAX_K:
            return recover_grid(self.local_decoder, order.restore_full(values), NATIVE)
        if order.name == Order.O0:
            return recover_grid(self.local_decoder, values, NATIVE)
        return recover_groups(self.local_decoder, order.recovery_groups(values), NATIVE)

    @torch.inference_mode()
    def reconstruct(self, features: Tensor) -> Tensor:
        return self.rgb_decoder(features, NATIVE)

    @torch.inference_mode()
    def decode(self, values: Tensor, order: RankOrder) -> Tensor:
        return self.reconstruct(self.recover(values, order))


def stream_frames(model: HLATok, frames: Iterable[tuple[Tensor, int]],
                  order: RankOrder) -> Iterator[tuple[Tensor, Tensor]]:
    """Emit each frame as soon as its input and selected budget are available."""
    receiver = ContinuousReceiver.from_model(model)
    for source, budget in frames:
        prefix = truncate(order.apply(encode_continuous(model, source)), budget)
        yield prefix, receiver.decode(prefix, order)
