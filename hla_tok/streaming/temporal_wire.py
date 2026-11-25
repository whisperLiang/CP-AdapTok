"""HLT v1 frame and record serialization. Little endian, with independent CRCs."""

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from zlib import crc32

from hla_tok.streaming.continuous import MAX_K, NATIVE
from hla_tok.temporal.layout import LAYOUT

MAGIC = b"HLT"
VERSION = 1
NO_REFERENCE = 0xFFFFFFFF
NO_DIGEST = bytes(32)
_CRC = Struct("<I")
_HEADER = Struct("<3sBBHIIHHHBH32s32s")
_RECORD = Struct("<HIBHH")
HEADER_BYTES = _HEADER.size + _CRC.size
RECORD_BYTES = _RECORD.size + _CRC.size
BASE_FLAG = 0x80
MAX_BLOCK = 256


class Mode(IntEnum):
    RAW4 = 0
    RAW8 = 1
    ZERO = 2
    BITMAP4 = 3


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    reference_id: int
    profile_id: int
    base_records: int
    binding: bytes
    reference_digest: bytes

    @property
    def intra(self) -> bool:
        return self.reference_id == NO_REFERENCE

    def validate(self) -> None:
        if not 0 <= self.frame_id < NO_REFERENCE or not 0 < self.profile_id <= 65535:
            raise ValueError("invalid frame or profile ID")
        if not 0 < self.base_records <= 65535 or len(self.binding) != 32 or len(self.reference_digest) != 32:
            raise ValueError("invalid base boundary or SHA-256 digest")
        if self.intra:
            if self.reference_digest != NO_DIGEST:
                raise ValueError("I frame must carry the empty reference digest")
        elif not 0 <= self.reference_id < self.frame_id:
            raise ValueError("P frame must address an earlier reference frame")

    def serialize(self) -> bytes:
        self.validate()
        fields = _HEADER.pack(MAGIC, VERSION, int(not self.intra), self.profile_id, self.frame_id,
                              self.reference_id, NATIVE.height, NATIVE.width, MAX_K, LAYOUT.components,
                              self.base_records, self.binding, self.reference_digest)
        return fields + _CRC.pack(crc32(fields))


def open_header(raw: bytes | memoryview) -> FrameHeader:
    if len(raw) < HEADER_BYTES:
        raise ValueError("truncated HLT frame header")
    expected, = _CRC.unpack_from(raw, _HEADER.size)
    if crc32(raw[:_HEADER.size]) != expected:
        raise ValueError("invalid HLT header CRC")
    magic, version, kind, profile, frame, reference, height, width, tokens, width_d, base, binding, digest = (
        _HEADER.unpack_from(raw))
    if (magic, version, height, width, tokens, width_d) != (
        MAGIC, VERSION, NATIVE.height, NATIVE.width, MAX_K, LAYOUT.components
    ):
        raise ValueError("invalid HLT version or native geometry")
    header = FrameHeader(frame, reference, profile, base, binding, digest)
    header.validate()
    if kind != int(not header.intra):
        raise ValueError("invalid HLT frame type")
    return header


def record_bytes(index: int, start: int, count: int, base: bool, mode: int, payload: bytes) -> bytes:
    fields = _RECORD.pack(index, start, int(mode) | (BASE_FLAG if base else 0), count, len(payload))
    return fields + payload + _CRC.pack(crc32(payload, crc32(fields)))


@dataclass(frozen=True)
class Record:
    index: int
    start: int
    count: int
    base: bool
    mode: Mode
    payload: bytes | memoryview

    def serialize(self) -> bytes:
        if not 1 <= self.count <= MAX_BLOCK or len(self.payload) > MAX_BLOCK:
            raise ValueError("invalid HLT record length")
        return record_bytes(self.index, self.start, self.count, self.base, self.mode, bytes(self.payload))


def open_record(raw: bytes | memoryview, offset: int = 0) -> tuple[Record, int]:
    if offset < 0 or len(raw) - offset < RECORD_BYTES:
        raise ValueError("truncated HLT record header")
    index, start, flags, count, size = _RECORD.unpack_from(raw, offset)
    start_payload = offset + _RECORD.size
    end_payload = start_payload + size
    stop = end_payload + _CRC.size
    if not 1 <= count <= MAX_BLOCK or size > count or stop > len(raw):
        raise ValueError("invalid HLT record length")
    expected, = _CRC.unpack_from(raw, end_payload)
    view = memoryview(raw)
    if crc32(view[offset:end_payload]) != expected:
        raise ValueError("invalid HLT record CRC")
    mode = Mode(flags & ~BASE_FLAG)
    return Record(index, start, count, bool(flags & BASE_FLAG), mode, view[start_payload:end_payload]), stop
