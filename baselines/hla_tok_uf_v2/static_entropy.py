"""Frozen enhancement histograms with CC-S deterministic integer allocation."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from baselines.hla_tok_uf_v2.entropy import PRECISION, RADIUS, EntropyTables

ALPHABET = 2 * RADIUS + 2


def integer_cdf(counts: NDArray[np.int64]) -> NDArray[np.int32]:
    """Reserve one count per symbol, then break largest-remainder ties by symbol ID."""
    if (counts.ndim != 2 or counts.shape[1] != ALPHABET or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or np.any(counts > 10**12)):
        raise ValueError("expected bounded enhancement histogram counts")
    weights = counts.astype(np.int64) + 1
    numerator = (PRECISION - ALPHABET) * weights
    quotient, remain = np.divmod(numerator, weights.sum(axis=1, keepdims=True))
    frequencies = quotient + 1
    order = np.argsort(-remain, axis=1, kind="stable")
    row, rank = np.where(np.arange(ALPHABET) < (PRECISION - frequencies.sum(axis=1))[:, None])
    frequencies[row, order[row, rank]] += 1
    return np.pad(frequencies.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0)))


@dataclass
class Histogram:
    tile_channels: int

    def __post_init__(self):
        self.counts: dict[bool, NDArray[np.int64]] = {}

    def observe(self, intra: bool, symbols: NDArray[np.int8]) -> None:
        if symbols.ndim != 4 or symbols.shape[0] != 2 or np.any(symbols == -128):
            raise ValueError("expected two signed refinement planes")
        channels = self.tile_channels
        bands = (symbols.shape[1] + channels - 1) // channels
        counts = self.counts.setdefault(intra, np.zeros((2, bands, ALPHABET), dtype=np.int64))
        signed = symbols.astype(np.int16)
        binned = np.minimum(np.abs(signed) * 2 - (signed > 0), ALPHABET - 1)
        for band in range(bands):
            region = binned[:, band * channels:(band + 1) * channels]
            for level in range(2):
                counts[level, band] += np.bincount(region[level].ravel(), minlength=ALPHABET)

    def fit(self, minimum: int = 4096) -> "StaticAsset":
        tables = {}
        for intra, histogram in self.counts.items():
            shrink = histogram.sum(-1, keepdims=True) < minimum
            parent = histogram.sum(axis=1, keepdims=True)
            merged = np.where(shrink, parent, histogram).reshape(-1, ALPHABET)
            tables[intra] = EntropyTables(tuple(tuple(map(int, row)) for row in integer_cdf(merged)))
        return StaticAsset(tables)


def _name(intra: bool) -> str:
    return "I" if intra else "P"


@dataclass(frozen=True)
class StaticAsset:
    tables: dict[bool, EntropyTables]

    @property
    def digest(self) -> str:
        if set(self.tables) != {True, False}:
            raise ValueError("a complete CDF asset needs independent I and P tables")
        buffer = b"UFV2 CDF v1\x00" + b"".join(
            np.asarray(self.tables[intra].cdf, dtype="<u4").tobytes() for intra in (True, False))
        return sha256(buffer).hexdigest()

    def as_json(self) -> dict:
        return {"sha256": self.digest, "cdf": {_name(i): table.cdf for i, table in self.tables.items()},
                "precision": PRECISION, "alphabet": "0,+1,-1,...,+8,-8,escape",
                "binary_table_bytes": sum(4 * len(row) for table in self.tables.values() for row in table.cdf)}

    @classmethod
    def load(cls, path: Path) -> "StaticAsset":
        import json

        data = json.loads(path.read_text())
        asset = cls({intra: EntropyTables(tuple(map(tuple, data["cdf"][_name(intra)]))) for intra in (True, False)})
        if data["sha256"] != asset.digest or data["precision"] != PRECISION:
            raise ValueError("static entropy table identity differs")
        return asset
