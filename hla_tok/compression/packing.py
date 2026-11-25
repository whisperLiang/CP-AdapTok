"""Canonical little-endian, token-aligned scalar words."""

import numpy as np
import torch
from torch import Tensor


def scalar_bounds(bits: int) -> tuple[int, int]:
    if bits not in (4, 6, 8):
        raise ValueError("expected 4, 6 or 8 bits")
    half = 1 << (bits - 1)
    return -half, half - 1


def _pad_columns(values: np.ndarray, multiple: int) -> np.ndarray:
    missing = -values.shape[1] % multiple
    return np.pad(values, ((0, 0), (0, missing))) if missing else values


def _pack4(u: np.ndarray) -> np.ndarray:
    u = _pad_columns(u, 2)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def _pack6(u: np.ndarray) -> np.ndarray:
    words = _pad_columns(u, 4).reshape(u.shape[0], -1, 4)
    output = np.empty((*words.shape[:2], 3), dtype=np.uint8)
    a, b, c, d = (words[..., i] for i in range(4))
    output[..., 0] = a | (b << 6)
    output[..., 1] = (b >> 2) | (c << 4)
    output[..., 2] = (c >> 4) | (d << 2)
    return output.reshape(u.shape[0], -1)


def _unpack4(data: np.ndarray) -> np.ndarray:
    output = np.empty((data.shape[0], data.shape[1] * 2), dtype=np.uint8)
    output[:, 0::2] = data & 15
    output[:, 1::2] = data >> 4
    return output


def _unpack6(data: np.ndarray) -> np.ndarray:
    words = _pad_columns(data, 3).reshape(data.shape[0], -1, 3)
    output = np.empty((*words.shape[:2], 4), dtype=np.uint8)
    a, b, c = (words[..., i] for i in range(3))
    output[..., 0] = a & 63
    output[..., 1] = (a >> 6) | ((b & 15) << 2)
    output[..., 2] = (b >> 4) | ((c & 3) << 4)
    output[..., 3] = c >> 2
    return output.reshape(data.shape[0], -1)


_PACK = {4: _pack4, 6: _pack6}
_UNPACK = {4: _unpack4, 6: _unpack6}


def _words(values: Tensor) -> np.ndarray:
    if values.device.type != "cpu" or values.dtype != torch.int8 or values.ndim != 2 or not values.numel():
        raise ValueError("expected nonempty signed CPU int8 [count, dimension]")
    return values.contiguous().numpy()


def pack_signed(values: Tensor, bits: int) -> bytes:
    low, high = scalar_bounds(bits)
    words = _words(values)
    if bits == 8:
        return words.tobytes()
    if words.min() < low or words.max() > high:
        raise ValueError("integer is outside the signed wire range")
    unsigned = words.view(np.uint8) & ((1 << bits) - 1)
    stride = (words.shape[1] * bits + 7) // 8
    return _PACK[bits](unsigned)[:, :stride].tobytes()


def _byte_rows(payload: bytes | memoryview, bits: int, dimension: int) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("expected positive token dimension")
    stride = (bits * dimension + 7) // 8
    if not len(payload) or len(payload) % stride:
        raise ValueError("payload does not contain complete tokens")
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(-1, stride)
    used = bits * dimension % 8
    if used and np.any(rows[:, -1] >> used):
        raise ValueError("unused token padding bits must be zero")
    return rows


def unpack_signed(payload: bytes | memoryview, bits: int, dimension: int) -> Tensor:
    _, high = scalar_bounds(bits)
    data = _byte_rows(payload, bits, dimension)
    if bits == 8:
        return torch.from_numpy(data.view(np.int8).copy())
    unsigned = _UNPACK[bits](data)[:, :dimension]
    sign = high + 1
    return torch.from_numpy(((unsigned ^ sign) - sign).view(np.int8))


def pack_bf16(values: Tensor) -> bytes:
    if values.device.type != "cpu" or values.dtype != torch.bfloat16:
        raise ValueError("expected CPU BF16 vectors")
    return values.detach().contiguous().view(torch.int16).numpy().astype("<i2", copy=False).tobytes()


def unpack_bf16(payload: bytes | memoryview, dimension: int) -> Tensor:
    words = np.frombuffer(payload, dtype="<i2").astype("=i2", copy=True)
    return torch.from_numpy(words).view(torch.bfloat16).reshape(-1, dimension)
