"""Length-delimited, budget-independent application access units."""

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from struct import Struct
from zlib import crc32

from hla_tok.uf.base import BaseFrame, FrameInfo

MAGIC = b"HUF1"
BASE_HEADER = Struct("<4s8sIHHBBI")
RECORD_HEADER = Struct("<IBBH")
CHECKSUM = Struct("<I")
APP_BASE_OVERHEAD = BASE_HEADER.size + CHECKSUM.size
RECORD_OVERHEAD = RECORD_HEADER.size + CHECKSUM.size
MAX_BASE_BYTES = 8 << 20
MAX_FRAME_BYTES = 16 << 20
MIN_RANS_BYTES = 4


class WireError(ValueError):
    pass


class BaseBudgetExceeded(WireError):
    def __init__(self, required: int, budget: int):
        super().__init__(f"BASE_BUDGET_EXCEEDED: {required} bytes required for {budget}-byte budget")


@dataclass(frozen=True, slots=True)
class Enhancement:
    group: int
    level: int
    cdf: int
    payload: bytes

    def __post_init__(self) -> None:
        _check_record(self.group, self.level, self.cdf, len(self.payload))


def _check_record(group: int, level: int, cdf: int, size: int) -> None:
    if not 0 <= group < 2**32 or level not in (0, 1) or not 0 <= cdf <= 255:
        raise WireError("invalid enhancement record fields")
    if not MIN_RANS_BYTES <= size <= 65535:
        raise WireError("invalid enhancement payload length")


class RefinementOrder:
    def __init__(self) -> None:
        self.levels: dict[int, int] = {}

    def add(self, group: int, level: int) -> None:
        expected = self.levels.get(group, 0)
        if level < expected:
            raise WireError("duplicate enhancement")
        if level > expected:
            raise WireError("missing refinement dependency")
        self.levels[group] = level + 1


def _protected(header: bytes, payload: bytes) -> bytes:
    return header + CHECKSUM.pack(crc32(payload, crc32(header))) + payload


def pack_base(base: BaseFrame, tag: bytes) -> bytes:
    info, payload = base.info, base.payload
    if len(tag) != 8 or not MIN_RANS_BYTES <= len(payload) <= MAX_BASE_BYTES:
        raise WireError("invalid model tag or base length")
    flags = int(info.intra) | (int(info.reset) << 1) | ((info.parallel - 1) << 2)
    header = BASE_HEADER.pack(MAGIC, tag, info.index, info.height, info.width, info.qp, flags, len(payload))
    return _protected(header, payload)


def pack_enhancement(record: Enhancement) -> bytes:
    return _protected(RECORD_HEADER.pack(record.group, record.level, record.cdf, len(record.payload)),
                      record.payload)


@dataclass(frozen=True, slots=True)
class PrefixFamily:
    raw: bytes
    ends: tuple[int, ...]

    @classmethod
    def build(cls, base: BaseFrame, tag: bytes, records: Iterable[Enhancement],
              max_bytes: int = MAX_FRAME_BYTES) -> "PrefixFamily":
        packets = [pack_base(base, tag)]
        ends = [len(packets[0])]
        order = RefinementOrder()
        for record in records:
            order.add(record.group, record.level)
            packet = pack_enhancement(record)
            if ends[-1] + len(packet) > max_bytes:
                break
            packets.append(packet)
            ends.append(ends[-1] + len(packet))
        if ends[-1] > MAX_FRAME_BYTES:
            raise WireError("frame exceeds maximum application length")
        return cls(b"".join(packets), tuple(ends))

    @property
    def base_end(self) -> int:
        return self.ends[0]

    def prefix(self, budget: int) -> tuple[bytes, int]:
        if budget < self.base_end:
            raise BaseBudgetExceeded(self.base_end, budget)
        index = bisect_right(self.ends, budget) - 1
        return self.raw[:self.ends[index]], index


class FrameParser:
    def __init__(self, tag: bytes):
        if len(tag) != 8:
            raise ValueError("model tag must have exactly eight bytes")
        self.tag = tag
        self._buffer = bytearray()
        self._offset = 0
        self._received = 0
        self._has_base = False
        self._failed = False
        self._order = RefinementOrder()

    def feed(self, chunk: bytes) -> None:
        if self._failed or self._received + len(chunk) > MAX_FRAME_BYTES:
            self._failed = True
            raise WireError("frame rejected or exceeds maximum application length")
        self._buffer.extend(chunk)
        self._received += len(chunk)

    def _extract(self, header_size: int, length: int) -> bytes | None:
        start = self._offset
        header_end = start + header_size
        data_start = header_end + CHECKSUM.size
        end = data_start + length
        if end > len(self._buffer):
            return None
        checksum, = CHECKSUM.unpack_from(self._buffer, header_end)
        data = bytes(self._buffer[data_start:end])
        if checksum != crc32(data, crc32(self._buffer[start:header_end])):
            raise WireError("record checksum mismatch")
        self._offset = end
        return data

    def _base(self) -> BaseFrame | None:
        if len(self._buffer) - self._offset < APP_BASE_OVERHEAD:
            return None
        magic, tag, index, height, width, qp, flags, size = BASE_HEADER.unpack_from(self._buffer, self._offset)
        if magic != MAGIC or tag != self.tag:
            raise WireError("unknown wire version or unexpected model")
        if not MIN_RANS_BYTES <= size <= MAX_BASE_BYTES or flags & ~31:
            raise WireError("invalid base record length or flags")
        payload = self._extract(BASE_HEADER.size, size)
        if payload is None:
            return None
        info = FrameInfo(index, height, width, qp, bool(flags & 1), (flags >> 2) + 1, bool(flags & 2))
        self._has_base = True
        return BaseFrame(info, payload)

    def _enhancement(self) -> Enhancement | None:
        if len(self._buffer) - self._offset < RECORD_OVERHEAD:
            return None
        group, level, cdf, size = RECORD_HEADER.unpack_from(self._buffer, self._offset)
        _check_record(group, level, cdf, size)
        payload = self._extract(RECORD_HEADER.size, size)
        if payload is None:
            return None
        self._order.add(group, level)
        return Enhancement(group, level, cdf, payload)

    def available(self) -> Iterator[BaseFrame | Enhancement]:
        if self._failed:
            raise WireError("frame already rejected")
        try:
            while (record := self._enhancement() if self._has_base else self._base()) is not None:
                yield record
        except ValueError as error:
            self._failed = True
            raise WireError(str(error)) from error
        finally:
            del self._buffer[:self._offset]
            self._offset = 0

    def finish(self) -> None:
        if self._failed or not self._has_base or self._buffer:
            raise WireError("incomplete or rejected application frame")
