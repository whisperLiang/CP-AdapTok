"""Frozen groupwise FSQ and a two-level integer payload contract."""

import math
from dataclasses import dataclass
from struct import Struct

import torch

from aip_vtok.c1.tokens import FRAME_HEADER, PACKET_TOKENS
from aip_vtok.diagnostics.diag0.sparse import decode_fsq, encode_fsq
from aip_vtok.tokenizer.fsq import FSQ

ORIGINAL = (8, 8, 8, 5, 5, 5)
CARDINALITY = math.prod(ORIGINAL)
CODE_BITS = 16
FP32_GROUP_BITS = 32 * len(ORIGINAL)
# First raster slot, symbol count, payload bytes, level (1 or 2).
LEVEL_HEADER = Struct("<IHHB")


class GroupFSQ(FSQ):
    """Reuse the sealed FP32 centers and exact mixed-radix arithmetic."""

    def __init__(self, levels, device):
        super().__init__()
        dims = torch.tensor(levels, dtype=torch.long, device=device)[:, None, None, :]
        self.levels = dims
        self.radix = torch.cat((torch.ones_like(dims[..., :1]), dims.cumprod(-1)[..., :-1]), dim=-1)
        self.half_width = dims // 2


class Stage:
    def __init__(self, levels, scales, mu=0):
        self.levels = tuple(map(tuple, levels))
        if any(len(dims) != 6 or min(dims) < 2 or math.prod(dims) != CARDINALITY for dims in self.levels):
            raise ValueError("each FSQ group must have six dimensions and exactly 64,000 codewords")
        if scales.shape != (len(self.levels), 6) or not (torch.isfinite(scales) & (scales > 0)).all():
            raise ValueError("FSQ requires finite positive per-group scales")
        if not isinstance(mu, (float, int)) or not math.isfinite(mu) or mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        self.scales = scales.detach().float()
        self.scale = self.scales[:, None, None, :]
        self.mu = mu
        self.log_mu = math.log1p(mu)
        self.fsq = GroupFSQ(self.levels, scales.device)

    def encode(self, values):
        if values.ndim != 5 or values.shape[1] != len(self.levels) or values.shape[-1] != 6:
            raise ValueError("FSQ input must have shape [B, groups, H, W, 6]")
        if self.mu == 0:
            return encode_fsq(self.fsq, values, self.scale)
        normalized = values.float() / self.scale
        compressed = normalized.sign() * torch.log1p(self.mu * normalized.abs()) / self.log_mu
        return encode_fsq(self.fsq, compressed, 1)

    def decode(self, ids):
        if ids.ndim != 4 or ids.shape[1] != len(self.levels) or ids.dtype != torch.int64:
            raise ValueError("FSQ IDs must be int64 with shape [B, groups, H, W]")
        if self.mu == 0:
            return decode_fsq(self.fsq, ids, self.scale)
        compressed = decode_fsq(self.fsq, ids, 1)
        return compressed.sign() * torch.expm1(self.log_mu * compressed.abs()) * (self.scale / self.mu)

    def record(self):
        return {"levels": self.levels, "scales": self.scales.tolist(), "mu": self.mu}

    @classmethod
    def load(cls, record, device):
        return cls(record["levels"], torch.tensor(record["scales"], device=device), record["mu"])


@dataclass(frozen=True)
class Quantizer:
    primary: Stage
    residual: Stage | None = None

    @property
    def stages(self):
        return (self.primary,) if self.residual is None else (self.primary, self.residual)

    @torch.no_grad()
    def encode(self, values):
        primary, saturated = self.primary.encode(values)
        if self.residual is None:
            return (primary,), (saturated,)
        residual, saturated_r = self.residual.encode(values - self.primary.decode(primary))
        return (primary, residual), (saturated, saturated_r)

    def decode(self, primary, residual=None):
        if primary is None:
            raise ValueError("Level 1 IDs are required to decode")
        coarse = self.primary.decode(primary)
        if residual is None:
            return coarse
        if self.residual is None:
            raise ValueError("single-stage FSQ cannot decode Level 2")
        if primary.shape != residual.shape:
            raise ValueError("Level 1 and Level 2 IDs must cover the same slots")
        return coarse + self.residual.decode(residual)


def wire_ledger(stage_tokens, bits_per_symbol, entropy_payload=None):
    """Frame + per-level packet headers under the declared diagnostic layout."""
    packet_count = sum(math.ceil(n / PACKET_TOKENS) for n in stage_tokens)
    header_bytes = FRAME_HEADER.size + LEVEL_HEADER.size * packet_count
    payload = sum(stage_tokens) * bits_per_symbol
    return {
        "packet_count": packet_count, "header_bytes": header_bytes, "payload_bits": payload,
        "raw_total_bytes": header_bytes + payload // 8,
        "entropy_est_bits": None if entropy_payload is None else header_bytes * 8 + entropy_payload,
        "real_bitstream": False,
    }


def dense_ledger(groups, stages, shape, marginal_entropy=None):
    area = math.prod(shape)
    tokens = groups * area
    widths = CODE_BITS if stages else FP32_GROUP_BITS
    streams = stages or 1
    entropy = None if marginal_entropy is None else marginal_entropy * tokens * streams
    return {
        **wire_ledger([tokens] * streams, widths, entropy),
        "payload_bits_per_position": groups * streams * widths,
        "symbols": stages * tokens,
    }
