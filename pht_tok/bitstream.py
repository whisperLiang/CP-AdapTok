"""PHT headers, checksummed entropy records and immutable byte prefixes."""

from bisect import bisect_right
from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from zlib import crc32

MAGIC = b"PHT1"
MAGIC_V2 = b"PHT2"
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
    LD_MASK = 4
    LD_INTRA = 5
    MASK_FINE = 6

    @property
    def backend(self) -> str:
        return "ld" if self in (Mode.LD_MASK, Mode.LD_INTRA) else "hts"

    @property
    def is_intra(self) -> bool:
        return self in (Mode.INTRA, Mode.LD_INTRA)

    @property
    def layers(self) -> int:
        return _LAYERS[self]

    @property
    def magic(self) -> bytes:
        return MAGIC if self <= Mode.INTRA else MAGIC_V2


_LAYERS = {Mode.MASK: 4, Mode.PRECISION: 4, Mode.INTRA: 1,
           Mode.LD_MASK: 8, Mode.LD_INTRA: 1, Mode.MASK_FINE: 8}


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
        if self.mode.is_intra:
            if (self.parent_id, self.valid_frames, self.k_ref, self.reset_memory) != (NO_REFERENCE, 1, 1, False):
                raise ValueError("invalid independent I-frame")
            return
        frame_limit = 1 if self.mode.backend == "ld" else 8
        if not (
            0 <= self.parent_id < self.chunk_id
            and 1 <= self.valid_frames <= frame_limit
            and 1 <= self.k_ref <= self.mode.layers
        ):
            raise ValueError("invalid progressive chunk dependencies")

    @property
    def record_count(self) -> int:
        return self.mode.layers + int(not self.mode.is_intra)

    def record_checksum(self, payload: bytes, index: int) -> int:
        return crc32(payload, index + 1 if self.mode.magic == MAGIC_V2 else 0)

    def pack(self) -> bytes:
        flags = int(self.mode) | (int(self.reset_memory) << 7)
        payload = HEADER.pack(self.mode.magic, self.model, self.chunk_id, self.parent_id, self.reference_version,
                              self.height, self.width, self.qp, self.valid_frames, flags, self.k_ref)
        return payload + CHECKSUM.pack(crc32(payload))

    @classmethod
    def unpack(cls, raw: bytes) -> "ChunkHeader":
        if len(raw) != HEADER_BYTES:
            raise CorruptStream("truncated or invalid header length")
        body = raw[:HEADER.size]
        if crc32(body) != CHECKSUM.unpack(raw[HEADER.size:])[0]:
            raise CorruptStream("chunk header checksum mismatch")
        magic, model, chunk, parent, version, height, width, qp, valid, flags, k_ref = HEADER.unpack(body)
        try:
            mode = Mode(flags & 127)
            if mode.magic != magic:
                raise ValueError("unsupported PHT format version for the stream mode")
            return cls(model, chunk, parent, version, height, width, qp, valid,
                       mode, k_ref, bool(flags & 128))
        except ValueError as error:
            raise CorruptStream(str(error)) from error


HEADER_BYTES = HEADER.size + CHECKSUM.size


@dataclass(frozen=True, slots=True)
class Chunk:
    data: bytes
    prefix_boundaries: tuple[int, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        header = ChunkHeader.unpack(self.data[:HEADER_BYTES])
        if len(self.sizes) != header.record_count:
            raise ValueError("invalid prefix index")
        ends = self._record_ends()
        if ends[int(not header.mode.is_intra):] != self.prefix_boundaries or ends[-1] != len(self.data):
            raise ValueError("invalid prefix index")

    def _record_ends(self) -> tuple[int, ...]:
        cursor = HEADER_BYTES
        ends = []
        for size in self.sizes:
            end = cursor + RECORD.size + size
            if not 4 <= size <= MAX_PAYLOAD or end > len(self.data):
                raise ValueError("prefix index exceeds the physical entropy record")
            if RECORD.unpack_from(self.data, cursor)[0] != size:
                raise ValueError("prefix index disagrees with record lengths")
            cursor = end
            ends.append(cursor)
        return tuple(ends)

    @classmethod
    def pack(cls, header: ChunkHeader, payloads: tuple[bytes, ...]) -> "Chunk":
        if len(payloads) != header.record_count:
            raise ValueError("all prescribed records are required")
        data = bytearray(header.pack())
        boundaries = []
        for index, payload in enumerate(payloads):
            if not 4 <= len(payload) <= MAX_PAYLOAD:
                raise ValueError("invalid entropy record length")
            data.extend(RECORD.pack(len(payload), header.record_checksum(payload, index)))
            data.extend(payload)
            if index or header.mode.is_intra:
                boundaries.append(len(data))
        return cls(bytes(data), tuple(boundaries), tuple(map(len, payloads)))

    def truncate_to_budget(self, max_bytes: int) -> bytes:
        count = bisect_right(self.prefix_boundaries, max_bytes)
        return self.data[:self.prefix_boundaries[count - 1]] if count else b""

    def prefix(self, *, max_bytes: int) -> bytes:
        return self.truncate_to_budget(max_bytes)

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
        self.received = 0
        self._history: bytearray | None = None

    @property
    def at_boundary(self) -> bool:
        return self.header is not None and self.records > int(not self.header.mode.is_intra) and not self.buffer

    @property
    def complete(self) -> bool:
        return self.header is not None and self.records == self.header.record_count

    def append_bytes(self, data: bytes, *, offset: int | None = None) -> tuple[bytes, ...]:
        if self.failed:
            raise CorruptStream("the chunk parser is invalidated")
        if offset is not None and self._is_replay(data, offset):
            return ()
        self.received += len(data)
        if self._history is not None:
            self._history.extend(data)
        self.buffer.extend(data)
        try:
            return self._read_records()
        except CorruptStream:
            self.failed = True
            raise

    def _is_replay(self, data: bytes, offset: int) -> bool:
        if not 0 <= offset <= self.received:
            raise CorruptStream("unexpected stream offset")
        if self._history is None and not self.received:
            self._history = bytearray()
        if offset == self.received:
            return False
        if self._history is None or data != self._history[offset:offset + len(data)]:
            raise CorruptStream("conflicting stream replay")
        return True

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
        assert self.header is not None
        if self.header.record_checksum(payload, self.records) != digest:
            raise CorruptStream("entropy record checksum mismatch")
        return payload, last
