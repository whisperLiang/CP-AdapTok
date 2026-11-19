"""Fixed-schedule six-dimensional symbols and receiver-only reconstruction."""

import math
from dataclasses import dataclass
from struct import Struct

import numpy as np
import torch

from aip_vtok.diagnostics.diag0.sparse import decode_fsq, encode_fsq
from aip_vtok.evaluation.fix_metrics import entropy_bits

from .scan import BUDGETS, GROUP_WIDTH, GROUPS, gaussian_fill

# Frame: magic, version, flags, QP, frame index, H, W.
# Packet: first slot index, token count, raw payload length. The final packet can be short.
FRAME_HEADER = Struct("<4sBBHIHH")
PACKET_HEADER = Struct("<IHH")
PACKET_TOKENS = 128
VERSIONS = ("cont", "fsq")


def grouped(ordered):
    return ordered.unflatten(1, (-1, GROUP_WIDTH))


def wire_cost(budget, entropy=None):
    endpoints = (0, *(k for k in BUDGETS if k < budget), budget)
    packets = sum(math.ceil((b - a) / PACKET_TOKENS) for a, b in zip(endpoints, endpoints[1:]))
    headers = FRAME_HEADER.size + packets * PACKET_HEADER.size
    token_bytes = budget * 2
    return {
        "packet_count": packets,
        "scan_table_bytes": 0,
        "frame_header_bytes": FRAME_HEADER.size,
        "packet_header_bytes_total": packets * PACKET_HEADER.size,
        "header_bytes": headers,
        "raw_token_bytes": token_bytes,
        "raw_total_bytes": token_bytes + headers,
        "entropy_total_est_bytes": None if entropy is None else headers + budget * entropy / 8,
        "real_bitstream": False,
    }


@dataclass(frozen=True)
class Packet:
    version: str
    payload: torch.Tensor
    saturated: torch.Tensor | None = None


@dataclass(frozen=True)
class Observations:
    values: torch.Tensor
    mask: torch.Tensor

    def ordered(self, sigma=None):
        values = self.values if sigma is None else gaussian_fill(self.values, self.mask, sigma)
        return values.flatten(1, 2)


def dense_roundtrip(fsq, values, scales):
    """Keep group, spatial position and six dimensions explicit throughout."""
    scale = scales[:, None, None, :]
    ids, saturation = encode_fsq(fsq, values.movedim(2, -1), scale)
    decoded = decode_fsq(fsq, ids, scale).movedim(-1, 2)
    return decoded, Packet("fsq", ids, saturation)


def send(fsq, values, scales, slots, version="fsq"):
    area = math.prod(values.shape[-2:])
    selected = values.movedim(2, -1).flatten(1, 3)[:, slots]
    if version == "cont":
        return Packet(version, selected)
    ids, saturation = encode_fsq(fsq, selected, scales[slots // area])
    return Packet(version, ids, saturation)


def receive(fsq, payload, scales, slots, shape, version):
    area = math.prod(shape)
    values = payload if version == "cont" else decode_fsq(fsq, payload, scales[slots // area])
    canvas = values.new_zeros((len(values), GROUPS * area, GROUP_WIDTH))
    canvas[:, slots] = values
    mask = torch.zeros((len(values), GROUPS * area), device=values.device, dtype=torch.bool)
    mask[:, slots] = True
    return Observations(canvas.reshape(len(values), GROUPS, *shape, GROUP_WIDTH).movedim(-1, 2),
                        mask.reshape(len(values), GROUPS, *shape))


def id_entropy(ids):
    counts = np.unique(ids.detach().cpu().numpy(), return_counts=True)[1]
    return entropy_bits(counts)


def fsq_counts(fsq, packet):
    """Integer per-group, per-dimension histogram and true pre-rounding saturation."""
    digits = fsq.ids_to_digits(packet.payload)
    dims = torch.arange(GROUP_WIDTH, device=digits.device).view(1, 1, 1, 1, -1)
    offsets = torch.arange(digits.shape[1], device=digits.device).view(1, -1, 1, 1, 1)
    bins = (offsets * GROUP_WIDTH + dims) * 8 + digits
    counts = torch.bincount(bins.flatten(), minlength=digits.shape[1] * GROUP_WIDTH * 8)
    saturated = packet.saturated.sum((0, 2, 3, 4))
    return counts.reshape(digits.shape[1], GROUP_WIDTH, 8).cpu().numpy(), saturated.cpu().numpy()
