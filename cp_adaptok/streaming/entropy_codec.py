"""Byte-renormalized rANS, 16-bit integer CDFs, independent terminated packets.

The initial model is static; BlockEntropyModel learns one parallel distribution per
packet position. CDFs are exported once on CPU, carried by the model artifact, and
identified in every frame header. No floating probability prediction runs at the receiver.
"""

import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PRECISION = 16
TOTAL = 1 << PRECISION
RANS_L = 1 << 23


def quantize_cdf(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not 2 <= len(weights) <= TOTAL // 2:
        raise ValueError("invalid alphabet")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("probabilities must be finite, nonnegative and nonempty")
    scaled = weights / weights.sum() * (TOTAL - len(weights))
    freq = np.floor(scaled).astype(np.int64) + 1
    remaining = TOTAL - int(freq.sum())
    order = np.lexsort((np.arange(len(weights)), -(scaled - np.floor(scaled))))
    freq[order[:remaining]] += 1
    return np.concatenate(([0], np.cumsum(freq))).tolist()


class ProgressiveEntropyCodec:
    def __init__(self, cdfs):
        self.cdfs = tuple(tuple(int(v) for v in cdf) for cdf in cdfs)
        if not self.cdfs:
            raise ValueError("no probability table")
        self.lookup = []
        for cdf in self.cdfs:
            self.validate_cdf(cdf)
            if len(cdf) != len(self.cdfs[0]):
                raise ValueError("all blocks must use the same alphabet")
            self.lookup.append(np.repeat(np.arange(len(cdf) - 1), np.diff(cdf)).tolist())
        raw = np.asarray(self.cdfs, dtype=">u4").tobytes()
        self.entropy_id = hashlib.sha256(raw).hexdigest()[:32]

    @staticmethod
    def validate_cdf(cdf):
        if not cdf:
            raise ValueError("empty CDF")
        if cdf[0] != 0 or cdf[-1] != TOTAL or any(b <= a for a, b in zip(cdf, cdf[1:])):
            raise ValueError("CDF must be strictly increasing from 0 to 65536")

    @classmethod
    def uniform(cls, codebook_size):
        return cls([quantize_cdf([1] * codebook_size)])

    def _table(self, block):
        if block < 0 or (len(self.cdfs) > 1 and block >= len(self.cdfs)):
            raise ValueError("unknown entropy block")
        index = 0 if len(self.cdfs) == 1 else block
        return self.cdfs[index], self.lookup[index]

    def encode(self, tokens, block):
        if not 1 <= len(tokens) <= 8192:
            raise ValueError("invalid token packet length")
        cdf, _ = self._table(block)
        state, out = RANS_L, bytearray()
        for symbol in reversed(tokens):
            if not 0 <= symbol < len(cdf) - 1:
                raise ValueError("symbol outside alphabet")
            start, freq = cdf[symbol], cdf[symbol + 1] - cdf[symbol]
            threshold = ((RANS_L >> PRECISION) << 8) * freq
            while state >= threshold:
                out.append(state & 255)
                state >>= 8
            state = ((state // freq) << PRECISION) + state % freq + start
        return struct.pack("<I", state) + bytes(reversed(out))

    def decode(self, payload, count, block):
        if len(payload) < 4 or not 1 <= count <= 8192:
            raise ValueError("invalid rANS packet")
        cdf, lookup = self._table(block)
        state = struct.unpack_from("<I", payload)[0]
        if not RANS_L <= state < RANS_L * 256:
            raise ValueError("invalid rANS initial state")
        position, output = 4, []
        for _ in range(count):
            slot = state & (TOTAL - 1)
            symbol = lookup[slot]
            output.append(symbol)
            state = (cdf[symbol + 1] - cdf[symbol]) * (state >> PRECISION) + slot - cdf[symbol]
            while state < RANS_L:
                if position >= len(payload):
                    raise ValueError("truncated rANS payload")
                state = (state << 8) | payload[position]
                position += 1
        if position != len(payload) or state != RANS_L:
            raise ValueError("rANS packet did not terminate exactly")
        return output


class BlockEntropyModel(nn.Module):
    """Lightweight learned, block-parallel probability model, with fixed integer export."""

    def __init__(self, codebook_size, blocks):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(blocks, codebook_size))

    def loss(self, ids, packet_tokens):
        positions = torch.arange(ids.shape[1], device=ids.device) // packet_tokens
        logits = self.logits[positions].unsqueeze(0).expand(ids.shape[0], -1, -1)
        return F.cross_entropy(logits.flatten(0, 1), ids.flatten()) / np.log(2)

    def export(self):
        probabilities = self.logits.detach().double().cpu().softmax(-1).numpy()
        return ProgressiveEntropyCodec([quantize_cdf(p) for p in probabilities])
