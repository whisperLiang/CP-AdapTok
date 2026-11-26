"""Independent, checksummed PHT v1 records and immutable byte prefixes."""

from bisect import bisect_right
from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from zlib import crc32

MAGIC = b"PHT1"
NO_REFERENCE = (1 << 32) - 1
MAX_PAYLOAD = 16 * 1024**2
HEADER = Struct(">4s8sIIIHHBBBB")
RECORD = Struct(">II")
CHECKSUM = Struct(">I")


class CorruptStream(ValueError):
    pass


class Mode(IntEnum):
    MASK = 1
    PRECISION = 2
    INTRA = 3


@dataclass(frozen=True, slots=True)
class ChunkHeader:
    model: bytes
    chunk_id: int
    parent_id: int
    reference_version: int
    height: int
    width: int
    qp: int
    valid_frames: int
    mode: Mode
    k_ref: int
    reset_memory: bool

    def __post_init__(self) -> None:
        if len(self.model) != 8 or not 0 <= self.chunk_id < NO_REFERENCE:
            raise ValueError("invalid model or chunk identity")
        fields = ((self.reference_version, 0, NO_REFERENCE - 1), (self.height, 1, 8192),
                  (self.width, 1, 8192), (self.qp, 0, 63))
        if any(not low <= value <= high for value, low, high in fields):
            raise ValueError("invalid stream geometry, version or QP")
        self._check_dependency()

    def _check_dependency(self) -> None:
        if self.mode == Mode.INTRA:
            if (self.parent_id, self.valid_frames, self.k_ref, self.reset_memory) != (NO_REFERENCE, 1, 1, False):
                raise ValueError("invalid independent I-frame")
            return
        if self.mode not in (Mode.MASK, Mode.PRECISION) or not (
            0 <= self.parent_id < self.chunk_id and 1 <= self.valid_frames <= 8 and 1 <= self.k_ref <= 4
        ):
            raise ValueError("invalid progressive chunk dependencies")

    @property
    def record_count(self) -> int:
        return 1 if self.mode == Mode.INTRA else 5

    def pack(self) -> bytes:
        flags = int(self.mode) | (int(self.reset_memory) << 7)
        payload = HEADER.pack(MAGIC, self.model, self.chunk_id, self.parent_id, self.reference_version,
                              self.height, self.width, self.qp, self.valid_frames, flags, self.k_ref)
        return payload + CHECKSUM.pack(crc32(payload))

    @classmethod
    def unpack(cls, raw: bytes) -> "ChunkHeader":
        body = raw[:HEADER.size]
        if crc32(body) != CHECKSUM.unpack(raw[HEADER.size:])[0]:
            raise CorruptStream("chunk header checksum mismatch")
        magic, model, chunk, parent, version, height, width, qp, valid, flags, k_ref = HEADER.unpack(body)
        if magic != MAGIC:
            raise CorruptStream("unsupported PHT format version")
        try:
            return cls(model, chunk, parent, version, height, width, qp, valid,
                       Mode(flags & 127), k_ref, bool(flags & 128))
        except ValueError as error:
            raise CorruptStream(str(error)) from error


HEADER_BYTES = HEADER.size + CHECKSUM.size


@dataclass(frozen=True, slots=True)
class Chunk:
    data: bytes
    prefix_boundaries: tuple[int, ...]
    sizes: tuple[int, ...]

    @classmethod
    def pack(cls, header: ChunkHeader, payloads: tuple[bytes, ...]) -> "Chunk":
        if len(payloads) != header.record_count:
            raise ValueError("all prescribed records are required")
        data = bytearray(header.pack())
        boundaries = []
        for index, payload in enumerate(payloads):
            if not 4 <= len(payload) <= MAX_PAYLOAD:
                raise ValueError("invalid entropy record length")
            data.extend(RECORD.pack(len(payload), crc32(payload)))
            data.extend(payload)
            if index or header.mode == Mode.INTRA:
                boundaries.append(len(data))
        return cls(bytes(data), tuple(boundaries), tuple(map(len, payloads)))

    def truncate_to_budget(self, max_bytes: int) -> bytes:
        count = bisect_right(self.prefix_boundaries, max_bytes)
        return self.data[:self.prefix_boundaries[count - 1]] if count else b""

    @property
    def ledger(self) -> dict:
        return {"header_bytes": HEADER_BYTES, "layer_header_bytes": len(self.sizes) * RECORD.size,
                "payload_bytes": list(self.sizes), "checksum_bytes_in_headers": (
                    CHECKSUM.size + len(self.sizes) * CHECKSUM.size
                ), "physical_bytes": len(self.data), "boundaries": list(self.prefix_boundaries)}


class StreamReader:
    """Keep only an unfinished header or payload between appends."""

    def __init__(self) -> None:
        self.header: ChunkHeader | None = None
        self.buffer = bytearray()
        self.records = 0
        self.failed = False

    @property
    def at_boundary(self) -> bool:
        return self.header is not None and self.records > int(self.header.mode != Mode.INTRA) and not self.buffer

    @property
    def complete(self) -> bool:
        return self.header is not None and self.records == self.header.record_count

    def append_bytes(self, data: bytes) -> tuple[bytes, ...]:
        if self.failed:
            raise CorruptStream("the chunk parser is invalidated")
        self.buffer.extend(data)
        try:
            return self._read_records()
        except CorruptStream:
            self.failed = True
            raise

    def _read_records(self) -> tuple[bytes, ...]:
        if self.header is None:
            if len(self.buffer) < HEADER_BYTES:
                return ()
            self.header = ChunkHeader.unpack(bytes(self.buffer[:HEADER_BYTES]))
            del self.buffer[:HEADER_BYTES]
        payloads = []
        offset = 0
        while self.records < self.header.record_count and len(self.buffer) - offset >= RECORD.size:
            record = self._payload_at(offset)
            if record is None:
                break
            payload, offset = record
            payloads.append(payload)
            self.records += 1
        del self.buffer[:offset]
        if self.complete and self.buffer:
            raise CorruptStream("unexpected bytes after the last record")
        return tuple(payloads)

    def _payload_at(self, offset: int) -> tuple[bytes, int] | None:
        size, digest = RECORD.unpack_from(self.buffer, offset)
        if not 4 <= size <= MAX_PAYLOAD:
            raise CorruptStream("invalid entropy record length")
        first, last = offset + RECORD.size, offset + RECORD.size + size
        if len(self.buffer) < last:
            return None
        payload = bytes(self.buffer[first:last])
        if crc32(payload) != digest:
            raise CorruptStream("entropy record checksum mismatch")
        return payload, last
