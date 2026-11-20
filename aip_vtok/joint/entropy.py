"""Factorized FSQ digit probabilities and independent integer rANS packets."""

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .transform import frame_type

PRECISION = 12
TOTAL = 1 << PRECISION
RANS_L = 1 << 23
STATE = struct.Struct("<I")


@dataclass(frozen=True)
class DigitCDF:
    radix: int
    frequency: tuple[int, ...]
    starts: tuple[int, ...]
    lookup: tuple[int, ...]


def integer_cdf(probs: list[float], radix: int) -> DigitCDF:
    remaining = TOTAL - len(probs)
    shares = [p / sum(probs) * remaining for p in probs]
    freq = [1 + math.floor(x) for x in shares]
    order = sorted(range(len(probs)), key=lambda i: (freq[i] - 1 - shares[i], i))
    for i in order[: TOTAL - sum(freq)]:
        freq[i] += 1
    starts: list[int] = []
    lookup: list[int] = []
    for i, count in enumerate(freq):
        starts.append(len(lookup))
        lookup.extend([i] * count)
    if len(lookup) != TOTAL:
        raise ValueError("integer digit CDF has invalid total")
    return DigitCDF(radix, tuple(freq), tuple(starts), tuple(lookup))


Tables = tuple[tuple[tuple[DigitCDF, ...], ...], ...]


class DigitEntropy(nn.Module):
    def __init__(self, levels: torch.Tensor):
        super().__init__()
        alphabet = torch.arange(int(levels.max()))
        self.register_buffer("valid", alphabet < levels[..., None])
        self.register_buffer("radix", torch.cat((torch.ones_like(levels[..., :1]), levels.cumprod(-1)[..., :-1]), -1))
        self.register_buffer("levels", levels)
        self.logits = nn.Parameter(-0.25 * (alphabet - levels[..., None] // 2).float().square())

    def log_probs(self) -> torch.Tensor:
        return self.logits.float().masked_fill(~self.valid, -torch.inf).log_softmax(-1)

    def bits(self, ids: torch.Tensor, groups: torch.Tensor, intra: bool,
             digit_position: torch.Tensor | None = None) -> torch.Tensor:
        kind = frame_type(intra)
        levels = self.levels[kind, groups]
        digits = ids[..., None] // self.radix[kind, groups] % levels
        costs = (-self.log_probs()[kind, groups] / math.log(2)).unsqueeze(0).expand(ids.shape[0], -1, -1, -1)
        hard = costs.gather(-1, digits.unsqueeze(-1)).squeeze(-1)
        if digit_position is not None:
            p = torch.minimum(digit_position.clamp_min(0), levels - 1)
            left = torch.minimum(p.detach().floor().long(), levels - 2)
            cost = costs.detach()
            lo = cost.gather(-1, left.unsqueeze(-1)).squeeze(-1)
            hi = cost.gather(-1, (left + 1).unsqueeze(-1)).squeeze(-1)
            proxy = torch.lerp(lo, hi, p - left)
            hard = hard + (proxy - proxy.detach())
        return hard.sum(-1)

    @torch.no_grad()
    def tables(self) -> Tables:
        probs = self.log_probs().exp().cpu().tolist()
        levels, radices = self.levels.cpu().tolist(), self.radix.cpu().tolist()
        return tuple(tuple(tuple(integer_cdf(p[:n], r) for p, n, r in zip(gp, gn, gr, strict=True))
                           for gp, gn, gr in zip(kp, kn, kr, strict=True))
                     for kp, kn, kr in zip(probs, levels, radices, strict=True))


def rans_encode(ids: Sequence[int], groups: Sequence[int], tables: tuple[tuple[DigitCDF, ...], ...]) -> bytes:
    state = RANS_L
    output = bytearray()
    for code, group in zip(reversed(ids), reversed(groups), strict=True):
        for cdf in reversed(tables[group]):
            digit = (code // cdf.radix) % len(cdf.frequency)
            frequency = cdf.frequency[digit]
            limit = ((RANS_L >> PRECISION) << 8) * frequency
            while state >= limit:
                output.append(state & 255)
                state >>= 8
            quotient, remainder = divmod(state, frequency)
            state = (quotient << PRECISION) + remainder + cdf.starts[digit]
    return STATE.pack(state) + bytes(reversed(output))


def rans_decode(payload: bytes, groups: Sequence[int], tables: tuple[tuple[DigitCDF, ...], ...]) -> tuple[int, ...]:
    if len(payload) < STATE.size:
        raise ValueError("truncated rANS state")
    state, = STATE.unpack_from(payload)
    if state < RANS_L or state >= (RANS_L << 8):
        raise ValueError("invalid rANS state")
    position, output = STATE.size, []
    for group in groups:
        code = 0
        for cdf in tables[group]:
            residue = state & (TOTAL - 1)
            digit = cdf.lookup[residue]
            code += digit * cdf.radix
            state = cdf.frequency[digit] * (state >> PRECISION) + residue - cdf.starts[digit]
            while state < RANS_L:
                if position == len(payload):
                    raise ValueError("truncated rANS renormalization")
                state = (state << 8) | payload[position]
                position += 1
        output.append(code)
    if position != len(payload) or state != RANS_L:
        raise ValueError("invalid rANS packet termination")
    return tuple(output)
