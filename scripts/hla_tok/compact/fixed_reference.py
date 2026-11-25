"""A measured evaluation envelope for historical Fixed13 index streams."""

from dataclasses import dataclass
from struct import Struct
from zlib import crc32

from hla_tok.streaming.fixed13 import TOKEN_BITS

_LAYOUT = Struct("<4sBBHHHII")
_CRC = Struct("<I")
HEADER_BYTES = _LAYOUT.size + _CRC.size


@dataclass(frozen=True)
class FixedWindow:
    start: int
    valid: int
    tiles: int
    groups: int
    count: int

    @property
    def tokens(self) -> int:
        return self.tiles * self.groups * self.count

    def check(self, length: int) -> None:
        if not (0 <= self.start < 2**32 and 1 <= self.valid <= 16 and 1 <= self.tiles <= 220
                and self.groups in (1, 4) and 0 < self.count <= 16320):
            raise ValueError("invalid Fixed13 window geometry")
        if length != (self.tokens * TOKEN_BITS + 7) // 8:
            raise ValueError("invalid Fixed13 window payload length")

    def packet(self, payload: bytes) -> bytes:
        self.check(len(payload))
        fields = _LAYOUT.pack(b"F13W", 1, self.valid, self.tiles, self.groups, self.count, self.start, len(payload))
        return fields + _CRC.pack(crc32(payload, crc32(fields))) + payload


def open_window(raw: bytes) -> tuple[FixedWindow, bytes]:
    if len(raw) < HEADER_BYTES:
        raise ValueError("truncated Fixed13 header")
    magic, version, valid, tiles, groups, count, start, length = _LAYOUT.unpack_from(raw)
    fields = FixedWindow(start, valid, tiles, groups, count)
    fields.check(length)
    crc, = _CRC.unpack_from(raw, _LAYOUT.size)
    payload = raw[HEADER_BYTES:]
    if (magic, version) != (b"F13W", 1) or length != len(payload) or crc != crc32(payload, crc32(raw[:_LAYOUT.size])):
        raise ValueError("invalid Fixed13 packet integrity")
    return fields, payload
