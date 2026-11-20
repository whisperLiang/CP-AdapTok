"""One stable learned order with constrained coverage and a mask surrogate."""

import hashlib
from collections.abc import Sequence

import torch
from torch import nn

from aip_vtok.c1.scan import SHAPE, halton_positions

from .tokenizer import GROUPS

HEIGHT, WIDTH = SHAPE
AREA = HEIGHT * WIDTH
SLOT_COUNT = GROUPS * AREA
BUDGETS = (510, 1024, 2048, 4096, 8192, 16320, 32640, SLOT_COUNT)
BASE_GROUPS = 4
TILES = 8
SCAN_VERSION = 1
REF_BUDGET = {True: 2048, False: 510}


def coverage(slots: torch.Tensor) -> torch.Tensor:
    group = slots // AREA
    y = slots % AREA // WIDTH
    x = slots % WIDTH
    return (group * TILES + y * TILES // HEIGHT) * TILES + x * TILES // WIDTH


def sp1_order(keys: dict[str, Sequence[Sequence[int]]]) -> torch.Tensor:
    """Use mean SP-1 rank for I/P and Halton order for newly available slots."""
    ranks = []
    positions = halton_positions(HEIGHT, WIDTH)
    for kind in ("I", "P"):
        old = [group * AREA + y * WIDTH + x for _, group, y, x in keys[kind]]
        rest = (g * AREA + p for p in positions for g in range(BASE_GROUPS))
        full = torch.tensor(list(dict.fromkeys([*old, *rest])))
        ranks.append(full.argsort())
    base = (ranks[0] + ranks[1]).argsort(stable=True).tolist()
    extra = [g * AREA + p for p in positions for g in range(BASE_GROUPS, GROUPS)]
    # Preserve the base prefix and give every fifth later slot to enhancements
    # until the original four-group lattice is exhausted.
    order = base[:BUDGETS[0]]
    for i, start in enumerate(range(BUDGETS[0], len(base), 4)):
        order.extend(base[start:start + 4])
        order.append(extra[i])
    return torch.tensor(order + extra[i + 1:])


def scan_digest(table: torch.Tensor) -> str:
    return hashlib.sha256(table.detach().cpu().numpy().astype("<u2").tobytes()).hexdigest()


class SharedScan(nn.Module):
    def __init__(self, initial: torch.Tensor, temperature: float = 0.02):
        super().__init__()
        slots = torch.arange(SLOT_COUNT)
        if initial.shape != slots.shape or not torch.equal(initial.cpu().sort().values, slots):
            raise ValueError("scan seed must be a complete slot permutation")
        if temperature <= 0:
            raise ValueError("scan temperature must be positive")
        self.score = nn.Parameter(1 - initial.cpu().argsort().float() / SLOT_COUNT)
        tile_ids = coverage(slots[:BASE_GROUPS * AREA])
        bands = [slots[:BASE_GROUPS * AREA][tile_ids == i] for i in range(BASE_GROUPS * TILES**2)]
        self.register_buffer("tiles", nn.utils.rnn.pad_sequence(bands, batch_first=True, padding_value=SLOT_COUNT),
                             persistent=False)
        self.register_buffer("slots", slots, persistent=False)
        self.register_buffer("locked_order", torch.full_like(slots, -1))
        self.frozen = False
        self.temperature = temperature

    @torch.no_grad()
    def table(self) -> torch.Tensor:
        if self.frozen:
            return self.locked_order
        raw = self.score.argsort(descending=True, stable=True)
        extended = torch.cat((self.score, self.score.new_full((1,), -torch.inf)))
        winners = self.tiles.gather(1, extended[self.tiles].argmax(1, keepdim=True)).flatten()
        mandatory = torch.zeros_like(self.score, dtype=torch.bool).index_fill(0, winners, True)[raw]
        base = (raw < BASE_GROUPS * AREA) & ~mandatory
        first = mandatory | (base & (base.cumsum(0) <= BUDGETS[0] - len(winners)))
        count = first.cumsum(0)
        dest = torch.where(first, count - 1, BUDGETS[0] + self.slots - count)
        return torch.empty_like(raw).scatter_(0, dest, raw)

    def mask(self, table: torch.Tensor, budget: int) -> torch.Tensor:
        if budget not in BUDGETS:
            raise ValueError("unknown training budget")
        hard = torch.zeros_like(self.score).index_fill_(0, table[:budget], 1)
        if not self.score.requires_grad or self.frozen or budget == SLOT_COUNT:
            return hard
        threshold = self.score.detach()[table[budget - 1]]
        soft = ((self.score - threshold) / self.temperature).sigmoid()
        if budget == BUDGETS[0]:
            soft = soft * (self.slots < BASE_GROUPS * AREA)
        return hard + (soft - soft.detach())

    @torch.no_grad()
    def freeze(self) -> torch.Tensor:
        self.locked_order.copy_(self.table())
        self.frozen = True
        self.score.requires_grad_(False)
        return self.locked_order

    def record(self) -> dict:
        return {"scan_sha256": scan_digest(self.table()), "scan_version": SCAN_VERSION,
                "slot_count": SLOT_COUNT, "prefix_boundaries": list(BUDGETS), "frozen": self.frozen}
