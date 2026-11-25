"""HLT arithmetic uses signed int16 lattice points with checked wider intermediates."""

import numpy as np

from hla_tok.compression.packing import scalar_bounds

LATTICE = np.iinfo(np.int16)


def checked_lattice(values: np.ndarray) -> np.ndarray:
    if values.min(initial=0) < LATTICE.min or values.max(initial=0) > LATTICE.max:
        raise OverflowError("temporal reference lattice exceeded its signed int16 range")
    return values.astype("<i2", copy=False)


def target_lattice(coefficients: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rounded = np.rint(coefficients / scale)
    if not np.isfinite(rounded).all():
        raise ValueError("nonfinite temporal target")
    return checked_lattice(rounded)


def round_divide(delta: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Round exact integer ratios to nearest with ties to even, for either sign."""
    quotient, remainder = np.divmod(delta, step)
    double = 2 * remainder
    return quotient + ((double > step) | ((double == step) & ((quotient & 1) != 0)))


def quantize_symbols(target: np.ndarray, prediction: np.ndarray | int, step: np.ndarray,
                     bits: int) -> tuple[np.ndarray, np.ndarray]:
    low, high = scalar_bounds(bits)
    code = round_divide(target.astype(np.int32) - prediction, step)
    return code.clip(low, high).astype(np.int8), (code < low) | (code > high)


def quantize(target: np.ndarray, prediction: np.ndarray | int, step: np.ndarray,
             bits: int) -> tuple[np.ndarray, int]:
    codes, clipped = quantize_symbols(target, prediction, step, bits)
    return codes, int(np.count_nonzero(clipped))


def reconstruct(prediction: np.ndarray | int, code: np.ndarray, step: np.ndarray) -> np.ndarray:
    return checked_lattice(prediction + step.astype(np.int64) * code)
