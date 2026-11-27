"""Integer CDFs for the pinned upstream rANS symbol order (0,+1,-1,...,escape)."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PRECISION = 65536
RADIUS = 8
MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)


def _mass(symbol: int, scale: float) -> float:
    return (1 - math.exp(-0.5 / scale) if symbol == 0 else
            (math.exp(-(symbol - 0.5) / scale) - math.exp(-(symbol + 0.5) / scale)) / 2)


def cdf_table(scale: float) -> tuple[int, ...]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid Laplace CDF scale")
    mass = [_mass(0, scale)] + [_mass(symbol, scale) for symbol in range(1, RADIUS + 1) for _ in range(2)]
    mass.append(math.exp(-(RADIUS + 0.5) / scale))
    remaining = PRECISION - len(mass)
    count = [int(p * remaining) + 1 for p in mass]
    count[0] += PRECISION - sum(count)
    return (0, *map(int, np.cumsum(count)))


@dataclass(frozen=True, slots=True)
class EntropyTables:
    cdf: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.cdf) <= 256 or any(not _valid(table) for table in self.cdf):
            raise ValueError("invalid integer-rANS CDF bank")

    @classmethod
    def from_scales(cls, scales: tuple[float, ...]) -> "EntropyTables":
        return cls(tuple(cdf_table(scale * factor) for scale in scales for factor in MULTIPLIERS))


def _valid(table: tuple[int, ...]) -> bool:
    return 3 <= len(table) <= 129 and table[0] == 0 and table[-1] == PRECISION and all(
        start < stop for start, stop in zip(table, table[1:]))


class SegmentCoder:
    def __init__(self, tables: EntropyTables):
        from hla_uf_rans import SegmentCodec

        self.codec = SegmentCodec(tables.cdf)

    def encode(self, symbols: NDArray[np.int8], cdf: int) -> bytes:
        return self.codec.encode(symbols, cdf)

    def decode(self, payload: bytes, cdf: int, count: int) -> NDArray[np.int8]:
        return self.codec.decode(payload, cdf, count)
