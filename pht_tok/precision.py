"""Refine one signed residual interval using the official HT-S Y CDF."""

from dataclasses import dataclass

import numpy as np

WIDTHS = (8, 4, 2, 1)
SYMBOL_OFFSET = 128
CDF_TOTAL = 65536


def permitted_symbols(symbols: np.ndarray, width: int) -> np.ndarray:
    return (symbols.astype(np.int16) + SYMBOL_OFFSET) // width * width - SYMBOL_OFFSET


def representative(low: np.ndarray, width: int) -> np.ndarray:
    """Choose the integer in the current interval closest to zero."""
    return np.clip(np.zeros_like(low), low, low + width - 1)


def _signed_weights(cdf: np.ndarray, length: int) -> np.ndarray:
    active = cdf[:length]
    if length < 3 or active[0] != 0 or active[-1] != CDF_TOTAL or np.any(np.diff(active) <= 0):
        raise ValueError("invalid official Gaussian CDF")
    symbols = np.arange(-128, 128, dtype=np.int32)
    ordered = 2 * np.abs(symbols) - (symbols > 0)
    frequencies = np.diff(active)
    escape = frequencies.size - 1
    tail = ordered >= escape
    output = np.empty(symbols.size, dtype=np.float64)
    output[~tail] = frequencies[ordered[~tail]]
    remainder = ordered[tail] - escape
    digits = (np.frexp(remainder)[1] + 1) // 2
    bypass = np.exp2(-2 * (digits + digits // 3 + 1))
    output[tail] = frequencies[-1] * bypass / bypass.sum()
    return output


def _quantized(weights: np.ndarray) -> np.ndarray:
    count = weights.shape[-1]
    scaled = weights / weights.sum(axis=-1, keepdims=True) * (CDF_TOTAL - count)
    freq = np.floor(scaled).astype(np.int32) + 1
    remaining = CDF_TOTAL - freq.sum(axis=-1)
    order = np.argsort(-(scaled - np.floor(scaled)), axis=-1, kind="stable")
    row = np.arange(len(freq))[:, None]
    freq[row, order] += np.arange(count) < remaining[:, None]
    return np.pad(np.cumsum(freq, axis=-1, dtype=np.int32), ((0, 0), (1, 0)))


@dataclass(frozen=True, slots=True)
class PrecisionTables:
    levels: tuple[np.ndarray, ...]

    @classmethod
    def from_original(cls, cdf: np.ndarray, lengths: np.ndarray) -> "PrecisionTables":
        weights = np.stack([_signed_weights(row, size) for row, size in zip(cdf, lengths, strict=True)])
        level_tables = [_quantized(weights.reshape(len(weights), -1, WIDTHS[0]).sum(axis=-1))]
        for width in WIDTHS[1:]:
            pairs = weights.reshape(len(weights) * 256 // (2 * width), 2, width).sum(axis=-1)
            level_tables.append(_quantized(pairs))
        return cls(tuple(level_tables))


def refinement_context(indexes: np.ndarray, parents: np.ndarray, width: int) -> np.ndarray:
    parent_count = 256 // (2 * width)
    return indexes.astype(np.uint16) * parent_count + parents.astype(np.uint16)
