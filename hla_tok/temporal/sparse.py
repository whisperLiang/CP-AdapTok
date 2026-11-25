"""Independent short-block modes. Bitmap and padding are physical wire data."""

from dataclasses import dataclass

import numpy as np

from hla_tok.compression.packing import _pack4, _unpack4
from hla_tok.streaming.temporal_wire import MAX_BLOCK, Mode


@dataclass(frozen=True)
class Block:
    mode: Mode
    payload: bytes
    padding_bits: int = 0


def _pack_bitmaps(values: np.ndarray) -> tuple[np.ndarray, bytes, np.ndarray]:
    nonzero = values != 0
    counts = nonzero.sum(axis=1)
    padding = counts % 2
    ends = (counts + padding).cumsum()
    symbols = np.zeros(ends[-1], dtype=np.uint8)
    indices = np.arange(counts.sum()) + np.repeat(padding.cumsum() - padding, counts)
    symbols[indices] = values[nonzero].view(np.uint8) & 15
    return np.packbits(nonzero, axis=1, bitorder="little"), _pack4(symbols[None]).tobytes(), ends // 2


def _modes(values: np.ndarray, bits: int, sparse: bool) -> tuple[np.ndarray, np.ndarray]:
    n, count = values.shape
    mode = np.full(n, bits == 8, dtype=np.uint8)
    lengths = np.full(n, (count * bits + 7) // 8, dtype=np.int32)
    if not sparse:
        return mode, lengths
    fits4 = np.ones(n, dtype=bool) if bits == 4 else ((values >= -8) & (values <= 7)).all(axis=1)
    nonzero = np.count_nonzero(values, axis=1)
    bitmap = (count + 7) // 8 + (nonzero + 1) // 2
    raw4 = (count + 1) // 2
    use_bitmap = fits4 & (bitmap < raw4)
    mode[fits4] = Mode.RAW4
    lengths[fits4] = raw4
    mode[use_bitmap] = Mode.BITMAP4
    lengths[use_bitmap] = bitmap[use_bitmap]
    mode[nonzero == 0] = Mode.ZERO
    lengths[nonzero == 0] = 0
    return mode, lengths


class Blocks:
    """Choose exact payload lengths before assembling the selected physical records."""

    def __init__(self, values: np.ndarray, bits: int, sparse: bool):
        self.values = values
        self.modes, self.lengths = _modes(values, bits, sparse)

    def payloads(self, stop: int):
        values = self.values[:stop]
        modes = self.modes[:stop]
        nibbles = _pack4(values.view(np.uint8) & 15) if (modes == Mode.RAW4).any() else None
        selected = modes == Mode.BITMAP4
        bitmap_batch = _pack_bitmaps(values[selected]) if selected.any() else None
        bitmap_index = bitmap_start = 0
        for index, mode in enumerate(modes.tolist()):
            if mode == Mode.RAW4:
                assert nibbles is not None
                payload = nibbles[index].tobytes()
            elif mode == Mode.BITMAP4:
                assert bitmap_batch is not None
                maps, symbols, ends = bitmap_batch
                bitmap_stop = int(ends[bitmap_index])
                payload = maps[bitmap_index].tobytes() + symbols[bitmap_start:bitmap_stop]
                bitmap_index += 1
                bitmap_start = bitmap_stop
            else:
                payload = values[index].tobytes() if mode == Mode.RAW8 else b""
            yield mode, payload


def pack_block(values: np.ndarray, bits: int, sparse: bool) -> Block:
    if values.dtype != np.int8 or values.ndim != 1 or not 1 <= values.size <= MAX_BLOCK:
        raise ValueError("expected a short signed int8 block")
    if bits not in (4, 8):
        raise ValueError("HLT supports 4-bit and 8-bit quantizers")
    if bits == 4 and (values.min() < -8 or values.max() > 7):
        raise ValueError("symbol is outside INT4")
    mode, payload = next(Blocks(values[None], bits, sparse).payloads(1))
    padding = 0
    if mode == Mode.RAW4:
        padding = (values.size % 2) * 4
    if mode == Mode.BITMAP4:
        padding = -values.size % 8 + (np.count_nonzero(values) % 2) * 4
    return Block(Mode(mode), payload, padding)


def unpack_block(mode: Mode, payload: bytes | memoryview, count: int) -> tuple[np.ndarray, int]:
    if not 1 <= count <= MAX_BLOCK or not isinstance(mode, Mode):
        raise ValueError("invalid short block length or mode")
    values, padding = decode_rows(mode, [payload], count)
    return values[0], padding


def _decode_bitmap_rows(payloads: list[bytes | memoryview], count: int) -> tuple[np.ndarray, int]:
    bitmap_bytes = (count + 7) // 8
    if any(len(raw) <= bitmap_bytes for raw in payloads):
        raise ValueError("incomplete BITMAP4 symbols")
    packed = np.frombuffer(b"".join(p[:bitmap_bytes] for p in payloads), dtype=np.uint8)
    packed = packed.reshape(len(payloads), bitmap_bytes)
    if count % 8 and (packed[:, -1] >> (count % 8)).any():
        raise ValueError("invalid bitmap padding")
    masks = np.unpackbits(packed, count=count, axis=1, bitorder="little").view(bool)
    written = masks.sum(axis=1)
    expected = (written + 1) // 2
    if not written.all() or not np.array_equal(expected, [len(raw) - bitmap_bytes for raw in payloads]):
        raise ValueError("BITMAP4 count does not match its symbols")
    words = np.frombuffer(b"".join(p[bitmap_bytes:] for p in payloads), dtype=np.uint8)
    nibbles = _unpack4(words[None]).reshape(-1)
    masked = np.ones(nibbles.size, dtype=bool)
    odd = (written % 2) != 0
    padding_at = (2 * expected.cumsum() - 1)[odd]
    if nibbles[padding_at].any():
        raise ValueError("nonzero BITMAP4 nibble padding")
    masked[padding_at] = False
    selected = nibbles[masked]
    if not selected.all():
        raise ValueError("BITMAP4 advertised a zero as nonzero")
    values = np.zeros((len(payloads), count), dtype=np.int8)
    values[masks] = ((selected ^ 8) - 8).view(np.int8)
    return values, (-count % 8) * len(payloads) + 4 * int(np.count_nonzero(odd))


def decode_rows(mode: Mode, payloads: list[bytes | memoryview], count: int) -> tuple[np.ndarray, int]:
    if mode == Mode.BITMAP4:
        return _decode_bitmap_rows(payloads, count)
    if mode == Mode.ZERO:
        if any(map(len, payloads)):
            raise ValueError("nonempty ZERO payload")
        return np.zeros((len(payloads), count), dtype=np.int8), 0
    stride = count if mode == Mode.RAW8 else (count + 1) // 2
    if any(len(raw) != stride for raw in payloads):
        raise ValueError("invalid HLT RAW payload length")
    encoded = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(len(payloads), stride)
    if mode == Mode.RAW8:
        return encoded.view(np.int8), 0
    if count % 2 and (encoded[:, -1] >> 4).any():
        raise ValueError("invalid HLT nibble padding")
    decoded = _unpack4(encoded)[:, :count]
    return ((decoded ^ 8) - 8).view(np.int8), len(payloads) * (count % 2) * 4
