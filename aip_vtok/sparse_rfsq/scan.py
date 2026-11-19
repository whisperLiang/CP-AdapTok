"""Static scan tables, exact integer prefixes and payload accounting."""

from collections import deque
from itertools import islice

import torch

from aip_vtok.c1.scan import BUDGETS, GROUPS, SHAPE, halton_positions, interleave
from aip_vtok.c1_followup.quantization import CODE_BITS, wire_ledger
from aip_vtok.c1_followup.sparse import SparsePlan

from .receiver import require_prefixes, sparse_values

ANCHORS, MAX_SYMBOLS = BUDGETS[0], BUDGETS[-1]
REFINEMENT_PCTS = (20, 30, 50)
POLICIES = ("SP-1", "SP-2", "SP-3")
LEGACY = "C1-A1"


def coarse_slots(weights):
    height, width = SHAPE
    cells = halton_positions(height, width)
    counts = [0] * GROUPS
    # Give every group a spatially uniform Halton anchor before using sensitivity quotas.
    for group, index in interleave((1,) * GROUPS, counts, len(cells), ANCHORS):
        yield 1, group, *divmod(cells[index], width)
    for group, index in interleave(weights, counts, len(cells), MAX_SYMBOLS - ANCHORS):
        yield 1, group, *divmod(cells[index], width)


def scan_table(name, weights, refinement_pct=0):
    if name not in POLICIES or len(weights) != GROUPS or min(weights) <= 0:
        raise ValueError("invalid scan policy or Train sensitivity weights")
    if refinement_pct not in (REFINEMENT_PCTS if name == "SP-3" else (0,)):
        raise ValueError("SP-3 requires a declared percentage of Level 2 symbols")
    coarse = iter(coarse_slots(weights))
    if name == "SP-1":
        return tuple(coarse)
    initial = 4096 if name == "SP-2" else ANCHORS
    pct = 100 if name == "SP-2" else refinement_pct
    return refine_slots(coarse, weights, initial, pct)


def refine_slots(coarse, weights, initial, pct):
    order = sorted(range(GROUPS), key=lambda g: (-weights[g], g))
    arrived = [deque() for _ in weights]
    keys = list(islice(coarse, initial))
    for _, group, y, x in keys:
        arrived[group].append((y, x))
    for index in range(1, MAX_SYMBOLS - initial + 1):
        if index * pct // 100 > (index - 1) * pct // 100:
            group = next(g for g in order if arrived[g])
            keys.append((2, group, *arrived[group].popleft()))
        else:
            slot = next(coarse)
            _, group, y, x = slot
            arrived[group].append((y, x))
            keys.append(slot)
    return tuple(keys)


def rates(keys, entropy, budgets=BUDGETS):
    """Fixed-16 and Train-empirical Shannon payload at each identical prefix."""
    counts = [0] * len(entropy)
    estimates, result = 0.0, {}
    for index, (level, group, _, _) in enumerate(keys, 1):
        counts[level - 1] += 1
        estimates += entropy[level - 1][group]
        if index in budgets:
            ledger = wire_ledger(counts, CODE_BITS, estimates)
            result[index] = {**ledger, "scan_table_bytes": 0, "entropy_payload_bits": estimates,
                             "level1_symbols": counts[0], "level2_symbols": sum(counts[1:])}
    return result


class Plan(SparsePlan):
    """The C1-Followup slot contract with a receiver that keeps both masks."""

    def encode(self, ids):
        count = self.schedule.shape[1]
        sent = self.encode_prefix(ids, count)
        require_prefixes(sent, (self.encode_prefix(ids, k) for k in BUDGETS if k < count))
        return sent

    def receive(self, quantizer, payload):
        if payload.ndim != 2 or payload.dtype != torch.int64:
            raise ValueError("sparse packets require [B, N] integer IDs")
        level, slot = self.prefix(payload.shape[1])
        stages = len(quantizer.stages)
        if stages == 1 and level.any():
            raise ValueError("received Level 2 for single-stage FSQ")
        area = self.groups * self.shape[0] * self.shape[1]
        offset = level * area + slot
        canvas = payload.new_zeros((len(payload), stages * area))
        mask = torch.zeros_like(canvas, dtype=torch.bool)
        canvas[:, offset] = payload
        mask[:, offset] = True
        shape = (len(payload), stages, self.groups, *self.shape)
        return sparse_values(quantizer, canvas.view(shape), mask.view(shape))
