"""Versioned little-endian packets for native C0 continuous vectors."""

from dataclasses import dataclass
from enum import Enum
from struct import Struct
from zlib import crc32

import numpy as np
import torch
from torch import Tensor

from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.continuous import MAX_K, TOKEN_DIM, Order, check_budget, check_vectors, truncate

_MAGIC = b"HLAC"
_VERSION = 1
_LAYOUT = Struct("<4sBBBBIHHHHI")
_CRC = Struct("<I")
HEADER_BYTES = _LAYOUT.size + _CRC.size
PIXELS_PER_FRAME = NATIVE_SHAPE[0] * NATIVE_SHAPE[1]


class WireDtype(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return _FORMATS[self][1]

    @property
    def bytes_per_token(self) -> int:
        return TOKEN_DIM * np.dtype(_FORMATS[self][3]).itemsize


_FORMATS = {
    WireDtype.FP32: (1, torch.float32, torch.int32, "<i4"),
    WireDtype.FP16: (2, torch.float16, torch.int16, "<i2"),
    WireDtype.BF16: (3, torch.bfloat16, torch.int16, "<i2"),
}
_CODES = {spec[0]: dtype for dtype, spec in _FORMATS.items()}


@dataclass(frozen=True)
class Packet:
    frame_id: int
    token_offset: int
    dtype: WireDtype
    order: Order
    values: Tensor

    @property
    def end(self) -> int:
        return self.token_offset + self.values.shape[1]


def _check_range(frame_id: int, offset: int, count: int) -> None:
    if type(frame_id) is not int or not 0 <= frame_id < 2**32:
        raise ValueError("frame ID must fit an unsigned 32-bit integer")
    if type(offset) is not int or not 0 <= offset < MAX_K:
        raise ValueError("invalid token offset")
    check_budget(count, MAX_K - offset)


def _crc(header, payload) -> int:
    return crc32(payload, crc32(header))


def serialize_cpu(values: Tensor, dtype: WireDtype, frame_id: int, token_offset: int = 0,
                  order: Order = Order.O0) -> bytes:
    check_vectors(values)
    code, wire_type, integer, endian = _FORMATS[dtype]
    if values.device.type != "cpu" or values.shape[0] != 1 or values.dtype != wire_type:
        raise ValueError("expected one CPU frame already cast to the requested wire dtype")
    count = values.shape[1]
    _check_range(frame_id, token_offset, count)
    payload = values.detach().contiguous().view(integer).numpy().astype(endian, copy=False).tobytes()
    fields = _LAYOUT.pack(_MAGIC, _VERSION, code, Order(order), TOKEN_DIM, frame_id, token_offset, count,
                          *NATIVE_SHAPE, len(payload))
    return fields + _CRC.pack(_crc(fields, payload)) + payload


def serialize_continuous_tokens(values: Tensor, dtype: WireDtype, frame_id: int, token_offset: int = 0,
                                order: Order = Order.O0) -> bytes:
    check_vectors(values)
    return serialize_cpu(values.to(dtype.torch_dtype).contiguous().cpu(), dtype, frame_id, token_offset, order)


def serialize_enhancement(full: Tensor, dtype: WireDtype, frame_id: int, start: int, end: int,
                          order: Order = Order.O0) -> bytes:
    _check_range(frame_id, start, end - start)
    return serialize_continuous_tokens(truncate(full, end)[:, start:], dtype, frame_id, start, order)


def _verify_length(raw: bytes, payload_size: int, count: int, dtype: WireDtype) -> memoryview:
    if payload_size != count * dtype.bytes_per_token or len(raw) != HEADER_BYTES + payload_size:
        raise ValueError("packet and declared payload lengths differ")
    payload = memoryview(raw)[HEADER_BYTES:]
    expected, = _CRC.unpack_from(raw, _LAYOUT.size)
    if _crc(memoryview(raw)[:_LAYOUT.size], payload) != expected:
        raise ValueError("continuous packet CRC32 mismatch")
    return payload


def deserialize_continuous_tokens(raw: bytes) -> Packet:
    if len(raw) < HEADER_BYTES:
        raise ValueError("truncated continuous packet header")
    magic, version, code, order, dim, frame, offset, count, height, width, size = _LAYOUT.unpack_from(raw)
    if (magic, version, dim, height, width) != (_MAGIC, _VERSION, TOKEN_DIM, *NATIVE_SHAPE):
        raise ValueError("unrecognized continuous protocol version or geometry")
    if code not in _CODES:
        raise ValueError("unrecognized wire dtype")
    dtype, rank_order = _CODES[code], Order(order)
    _check_range(frame, offset, count)
    payload = _verify_length(raw, size, count, dtype)
    raw_dtype = np.dtype(_FORMATS[dtype][3])
    words = np.frombuffer(payload, dtype=raw_dtype).astype(raw_dtype.newbyteorder("="), copy=True)
    values = torch.from_numpy(words).view(dtype.torch_dtype).reshape(1, count, TOKEN_DIM)
    return Packet(frame, offset, dtype, rank_order, values)


class FrameReceiver:
    """Reassemble one frame and publish only its contiguous, received prefix."""

    def __init__(self, frame_id: int, dtype: WireDtype, order: Order):
        self.identity = frame_id, dtype, order
        self._parts: list[Tensor] = []
        self._pending: dict[int, Packet] = {}
        self._count = 0
        self._committed = False

    def accept(self, raw: bytes) -> int:
        if self._committed:
            raise RuntimeError("frame delivery is already committed")
        return self._accept_packet(deserialize_continuous_tokens(raw))

    def _accept_packet(self, packet: Packet) -> int:
        if (packet.frame_id, packet.dtype, packet.order) != self.identity:
            raise ValueError("packet belongs to a different frame, dtype or Rank order identity")
        if packet.token_offset < self._count or any(
                other.token_offset < packet.end and packet.token_offset < other.end for other in self._pending.values()):
            raise ValueError("continuous packet overlap or duplicate")
        self._pending[packet.token_offset] = packet
        while self._count in self._pending:
            next_part = self._pending.pop(self._count)
            self._parts.append(next_part.values)
            self._count = next_part.end
        return self._count

    def prefix(self) -> Tensor | None:
        return torch.cat(self._parts, dim=1) if self._parts else None

    def commit(self) -> Tensor | None:
        self._committed = True
        self._pending.clear()
        return self.prefix()


def wire_rate(payload_bytes: int, total_bytes: int, frames: int) -> dict:
    if frames <= 0 or not 0 <= payload_bytes <= total_bytes:
        raise ValueError("invalid physical wire byte accounting")
    pixels = frames * PIXELS_PER_FRAME
    return {
        "original_pixels": pixels, "payload_bytes": payload_bytes, "header_bytes": total_bytes - payload_bytes,
        "side_information_bytes": 0, "total_bytes": total_bytes,
        "payload_bytes_per_frame": payload_bytes / frames, "total_bytes_per_frame": total_bytes / frames,
        "payload_bpp": 8 * payload_bytes / pixels, "total_bpp": 8 * total_bytes / pixels,
        **{f"mbps_{fps}fps": 8 * total_bytes * fps / (frames * 1e6) for fps in (24, 30, 60)},
    }


def max_budget_for_bytes(cap: int, dtype: WireDtype, *, include_header: bool = True) -> int:
    usable = cap - (HEADER_BYTES if include_header else 0)
    return max(0, min(MAX_K, usable // dtype.bytes_per_token))
