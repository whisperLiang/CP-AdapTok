"""Deterministic sparse support, discrete FSQ transport and byte accounting."""

import heapq
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from aip_vtok.evaluation.fix_metrics import entropy_bits

BUDGETS = (510, 1024, 2048, 4096, 8192)
REF_BUDGET = {True: 2048, False: 510}
GROUPS = 21
DEPTHS = np.array([0, 1, 2, 4, 8, 12], dtype=np.uint8)
LEVELS = (8, 8, 8, 5, 5, 5)
HEADER_BYTES = 24
QUANTILE = 0.995
QUANTILE_TANH_INPUT = 1.5


def fit_scale(samples):
    return (samples.abs().quantile(QUANTILE, dim=0) / QUANTILE_TANH_INPUT).clamp_min(1e-5)


def encode_fsq(fsq, values, scale):
    _, ids, saturated = fsq(values / scale, measure_saturation=True)
    return ids, saturated


def decode_fsq(fsq, ids, scale):
    codes = fsq.ids_to_codes(ids)
    extent = (fsq.levels.float() - 1) / 2
    offset = (fsq.levels % 2 == 0).float() / 2
    shift = torch.atanh(offset / extent)
    normalized = (codes * fsq.half_width + offset) / extent
    # Use the midpoint of each outer bin in tanh space to keep its inverse finite.
    bound = 1 - 0.25 / extent
    return (torch.atanh(normalized.clamp(-bound, bound)) - shift) * scale


def slot_energy(groups, weights):
    return (groups.square() * weights).sum(-1)


def top_slots(energy, maximum):
    return torch.argsort(energy.flatten(), descending=True, stable=True)[:maximum]


def upgrade_queue(energy, cell):
    height, width = energy.shape[:2]
    cells = energy.reshape(height // cell, cell, width // cell, cell, GROUPS).sum(axis=(1, 3))
    totals = cells.reshape(-1, GROUPS).cumsum(axis=1)[:, DEPTHS[1:] - 1]
    gains = np.diff(totals, axis=1, prepend=0) / np.diff(DEPTHS)
    queue = [(-gain, index, 0) for index, gain in enumerate(gains[:, 0])]
    heapq.heapify(queue)
    return cells.shape[:2], gains, queue


def grow_depth(queue, gains, depth, cells_left):
    deferred = []
    while queue and cells_left:
        item = heapq.heappop(queue)
        _, index, step = item
        cost = int(DEPTHS[step + 1] - DEPTHS[step])
        if cost > cells_left:
            deferred.append(item)
            continue
        depth[index] = DEPTHS[step + 1]
        cells_left -= cost
        if step + 1 < gains.shape[1]:
            heapq.heappush(queue, (-gains[index, step + 1], index, step + 1))
    for item in deferred:
        heapq.heappush(queue, item)
    return cells_left


def allocate_depth(energy, budgets, cell):
    """Greedy feasible upgrades at fixed milestones, with raster-order ties."""
    shape, gains, queue = upgrade_queue(energy, cell)
    depth = np.zeros(len(gains), dtype=np.uint8)
    spent, maps = 0, {}
    for budget in sorted(budgets):
        capacity = budget // cell**2
        spent = capacity - grow_depth(queue, gains, depth, capacity - spent)
        maps[budget] = depth.reshape(shape).copy()
    return maps


def prefix_slots(depth, cell):
    expanded = depth.repeat(cell, 0).repeat(cell, 1)
    return np.flatnonzero((np.arange(GROUPS) < expanded[..., None]).reshape(-1))


def pack_depth(depth):
    symbols = np.searchsorted(DEPTHS, depth.reshape(-1)).astype(np.uint8)
    if not np.isin(depth, DEPTHS).all():
        raise ValueError("depth has no 3-bit symbol")
    return np.packbits((symbols[:, None] >> np.arange(2, -1, -1) & 1).astype(np.uint8)).tobytes()


def unpack_depth(payload, shape):
    bit_count = int(np.prod(shape)) * 3
    if len(payload) != (bit_count + 7) // 8:
        raise ValueError("unexpected packed depth length")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), count=bit_count)
    symbols = (bits.reshape(-1, 3) * [4, 2, 1]).sum(axis=1)
    if (symbols >= len(DEPTHS)).any():
        raise ValueError("reserved 3-bit depth symbol")
    return DEPTHS[symbols].reshape(shape)


@dataclass(frozen=True)
class Symbols:
    values: torch.Tensor
    saturated: torch.Tensor | None

    @classmethod
    def encode(cls, values, scale, fsq, version):
        if version == "continuous":
            return cls(values, None)
        return cls(*encode_fsq(fsq, values, scale))

    def decode(self, scale, fsq):
        if self.values.is_floating_point():
            return self.values
        return decode_fsq(fsq, self.values, scale)


def symbol_histogram(fsq, symbols, slot_groups, group_count):
    digits = fsq.ids_to_digits(symbols.values)
    bins = slot_groups[:, None] * 48 + torch.arange(6, device=digits.device) * 8 + digits
    count = torch.bincount(bins.flatten(), minlength=group_count * 48).reshape(group_count, 6, 8)
    saturated = torch.zeros_like(count[:, 0, 0])
    saturated.scatter_add_(0, slot_groups, symbols.saturated.long().sum(1))
    return count.cpu().numpy(), saturated.cpu().numpy()


def entropy_bytes(histogram):
    return sum(entropy_bits(dim) for dim in histogram) * int(histogram[0].sum()) / 8


def depth_bytes(payload, shape):
    depth = unpack_depth(payload, shape)
    return {
        "R_depth_raw": len(payload),
        "R_depth_compressed_ref": len(zlib.compress(payload, level=9)),
        "R_depth_entropy_lb": depth.size * entropy_bits(np.bincount(depth.flatten())) / 8,
    }


def byte_ledger(budget, used, pixels, histogram=None, *, depth=None, support=0):
    raw_tokens = 2 * budget
    entropy_tokens = None if histogram is None else entropy_bytes(histogram)
    side_raw = (depth["R_depth_raw"] if depth else support) + HEADER_BYTES
    side_ref = (depth["R_depth_compressed_ref"] if depth else support) + HEADER_BYTES
    total_raw = raw_tokens + side_raw
    total_ref = None if entropy_tokens is None else entropy_tokens + side_ref
    return {
        "tokens_used": used,
        "R_tokens_raw": raw_tokens,
        "R_tokens_raw_used": 2 * used,
        "R_tokens_entropy_lb": entropy_tokens,
        "R_headers_est": HEADER_BYTES,
        "R_support_raw_est": support,
        **(depth or {}),
        "R_total_raw": total_raw,
        "R_total_entropy_ref": total_ref,
        "bpp_raw": total_raw * 8 / pixels,
        "bpp_entropy_ref": None if total_ref is None else total_ref * 8 / pixels,
        "side_info_raw_pct": 100 * side_raw / total_raw,
        "side_info_entropy_ref_pct": None if total_ref is None else 100 * side_ref / total_ref,
    }
