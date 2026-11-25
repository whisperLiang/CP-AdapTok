"""Compact packets for one native frame, with independently verifiable ranges."""

from dataclasses import dataclass
from struct import Struct
from zlib import crc32

from hla_tok.compression.profiles import PROFILES_BY_CODE, Profile
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.continuous import MAX_K, Order, check_budget

_MAGIC = b"HLCP"
_VERSION = 1
_FIELDS = Struct("<4sBBBBIHHHHI32s")
_CRC = Struct("<I")
HEADER_BYTES = _FIELDS.size + _CRC.size


@dataclass(frozen=True)
class WireIdentity:
    profile: Profile
    order: Order
    signature: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Profile) or not isinstance(self.order, Order) or len(self.signature) != 32:
            raise ValueError("invalid Compact profile, order or signature")


def _check_range(frame: int, offset: int, count: int) -> None:
    if type(frame) is not int or not 0 <= frame < 2**32:
        raise ValueError("frame ID must fit an unsigned 32-bit integer")
    if type(offset) is not int or not 0 <= offset < MAX_K:
        raise ValueError("invalid token offset")
    check_budget(count, MAX_K - offset)


@dataclass(frozen=True)
class Packet:
    identity: WireIdentity
    frame_id: int
    offset: int
    count: int
    payload: memoryview

    @property
    def end(self) -> int:
        return self.offset + self.count

    def ledger(self) -> dict[str, int]:
        payload_bytes = len(self.payload)
        return {"payload_bytes": payload_bytes, "header_bytes": HEADER_BYTES, "padding_bytes": 0,
                "side_information_bytes": 0, "total_bytes": HEADER_BYTES + payload_bytes}


@dataclass(frozen=True)
class CompactFrame:
    identity: WireIdentity
    frame_id: int
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or len(self.payload) % self.identity.profile.bytes_per_token:
            raise ValueError("frame payload does not contain complete tokens")
        _check_range(self.frame_id, 0, self.count)

    @property
    def count(self) -> int:
        return len(self.payload) // self.identity.profile.bytes_per_token

    def serialize_range(self, start: int, stop: int) -> bytes:
        check_budget(stop, self.count)
        count = stop - start
        _check_range(self.frame_id, start, count)
        profile, order, signature = self.identity.profile, self.identity.order, self.identity.signature
        stride = profile.bytes_per_token
        payload = self.payload[start * stride:stop * stride]
        header = _FIELDS.pack(_MAGIC, _VERSION, profile.code, order, profile.dimension,
                              self.frame_id, start, count, *NATIVE_SHAPE, len(payload), signature)
        return header + _CRC.pack(crc32(payload, crc32(header))) + payload


def deserialize(raw: bytes) -> Packet:
    if not isinstance(raw, bytes) or len(raw) < HEADER_BYTES:
        raise ValueError("truncated Compact packet header")
    magic, version, code, order, dim, frame, offset, count, h, w, size, tag = _FIELDS.unpack_from(raw)
    if (magic, version, h, w) != (_MAGIC, _VERSION, *NATIVE_SHAPE):
        raise ValueError("unrecognized Compact protocol or geometry")
    profile = PROFILES_BY_CODE.get(code)
    if profile is None or dim != profile.dimension:
        raise ValueError("unrecognized Compact profile or dimension")
    _check_range(frame, offset, count)
    if size != count * profile.bytes_per_token or len(raw) != HEADER_BYTES + size:
        raise ValueError("packet and declared payload lengths differ")
    view = memoryview(raw)
    payload = view[HEADER_BYTES:]
    expected, = _CRC.unpack_from(view, _FIELDS.size)
    if crc32(payload, crc32(view[:_FIELDS.size])) != expected:
        raise ValueError("Compact packet CRC32 mismatch")
    return Packet(WireIdentity(profile, Order(order), tag), frame, offset, count, payload)


class FrameAssembly:
    """Commit only the contiguous received prefix, even with missing enhancements."""

    def __init__(self, identity: WireIdentity, frame_id: int):
        _check_range(frame_id, 0, 1)
        self.identity = identity
        self.frame_id = frame_id
        self.count = 0
        self._committed = False
        self._pending: dict[int, Packet] = {}
        self._parts: list[memoryview] = []

    def accept(self, raw: bytes) -> int:
        if self._committed:
            raise RuntimeError("Compact delivery is already committed")
        packet = deserialize(raw)
        if (packet.identity, packet.frame_id) != (self.identity, self.frame_id):
            raise ValueError("packet identity differs from the configured frame")
        self._store(packet)
        return self.count

    def _store(self, packet: Packet) -> None:
        if packet.offset < self.count or any(packet.offset < other.end and packet.end > other.offset
                                              for other in self._pending.values()):
            raise ValueError("Compact packet overlap or duplicate")
        self._pending[packet.offset] = packet
        while self.count in self._pending:
            part = self._pending.pop(self.count)
            self._parts.append(part.payload)
            self.count = part.end

    def prefix(self) -> bytes | None:
        return b"".join(self._parts) if self.count else None

    def commit(self) -> bytes | None:
        self._committed = True
        self._pending.clear()
        return self.prefix()


def max_budget(cap: int, profile: Profile, *, payload_only: bool = False) -> int:
    if type(cap) is not int or cap < 0:
        raise ValueError("expected a nonnegative integer byte cap")
    available = cap if payload_only else cap - HEADER_BYTES
    return max(0, min(MAX_K, available // profile.bytes_per_token))
