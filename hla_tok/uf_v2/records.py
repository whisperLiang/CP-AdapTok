"""Batched HUF1 enhancement records and incremental integer accumulation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import BaseFrame
from hla_tok.uf.entropy import MULTIPLIERS, EntropyTables
from hla_tok.uf.refinement import GroupLayout
from hla_tok.uf.wire import (
    APP_BASE_OVERHEAD,
    BASE_HEADER,
    MAX_BASE_BYTES,
    MAX_FRAME_BYTES,
    MIN_RANS_BYTES,
    RECORD_OVERHEAD,
    BaseBudgetExceeded,
    Enhancement,
    FrameParser,
    PrefixFamily,
    WireError,
    pack_base,
    pack_enhancement,
)

if TYPE_CHECKING:
    from hla_uf_v2_records import Layout

RAW_FLAG = 128


@dataclass(frozen=True)
class Geometry:
    groups: GroupLayout
    native: Layout
    counts: NDArray[np.int32]


class BatchRecords:
    def __init__(self, scales: tuple[float, ...], segmentation: Segmentation,
                 tables: dict[bool, EntropyTables] | None = None, raw: bool = False,
                 sensitivity: dict[bool, NDArray[np.float64]] | None = None):
        from hla_uf_v2_records import Records

        self.scales: NDArray[np.float64] = np.asarray(scales, dtype=np.float64)[:, None]
        self.segmentation = segmentation
        self.static, self.raw, self.sensitivity = tables, raw, sensitivity
        legacy = EntropyTables.from_scales(scales)
        self.banks = {key: Records(table.cdf) for key, table in tables.items()} if tables else {
            True: Records(legacy.cdf), False: Records(legacy.cdf)}
        self.geometries: dict[tuple[int, int, int], Geometry] = {}

    def geometry(self, shape: tuple[int, int, int]) -> Geometry:
        from hla_uf_v2_records import Layout

        if shape not in self.geometries:
            layout = GroupLayout(shape, self.segmentation.tile, self.segmentation.channels)
            native = Layout(*shape, layout.tile, layout.channels)
            self.geometries[shape] = Geometry(layout, native, native.counts)
        return self.geometries[shape]

    def _priorities(self, statistics, counts, intra):
        magnitude = statistics[..., 0] / counts
        buckets = np.ones_like(magnitude, dtype=np.uint8)
        if self.segmentation.conditional:
            buckets = (np.rint(np.log2(np.maximum(0.125, magnitude) / self.scales)) + 1).clip(0, 3).astype(np.uint8)
        scale = self.scales * np.asarray(MULTIPLIERS)[buckets]
        bits = np.maximum(0.08, np.log2(2 * scale)) + magnitude / (scale * np.log(2))
        cost = RECORD_OVERHEAD + 4 + counts * bits / 8
        gain = statistics[..., 1] / np.array([[1], [4]])
        active = (gain != 0).astype(np.uint8)
        if self.sensitivity is not None:
            weight = self.sensitivity[intra]
            gain = gain * weight[np.arange(gain.shape[1]) % len(weight)]
        return gain / cost, active, buckets + np.array([[0], [len(MULTIPLIERS)]], dtype=np.uint8)

    def static_indexes(self, geometry: Geometry):
        bands = geometry.groups.grid[0]
        return (np.arange(len(geometry.groups)) % bands + np.arange(2)[:, None] * bands).astype(np.uint8)

    def family(self, base: BaseFrame, tag: bytes, coarse: Tensor, fine: Tensor, cap: int) -> PrefixFamily:
        return self.pack_symbols(base, tag, np.stack((coarse.numpy()[0], fine.numpy()[0])), cap)

    def pack_symbols(self, base: BaseFrame, tag: bytes, symbols: NDArray[np.int8], cap: int) -> PrefixFamily:
        prefix = pack_base(base, tag)
        if len(prefix) > cap:
            raise BaseBudgetExceeded(len(prefix), cap)
        if symbols.ndim != 4 or symbols.shape[0] != 2:
            raise ValueError("expected two CHW symbol planes")
        shape = (symbols.shape[1], symbols.shape[2], symbols.shape[3])
        geometry = self.geometry(shape)
        values, statistics = geometry.native.gather(np.ascontiguousarray(symbols))
        intra = base.info.intra
        scores, active, indexes = self._priorities(statistics, geometry.counts, intra)
        if self.static is not None:
            indexes = self.static_indexes(geometry)
        raw, ends = self.banks[intra].pack(geometry.native, values, scores, active, indexes,
                                          self.segmentation.ranked, cap - len(prefix), self.raw)
        return PrefixFamily(prefix + raw, (len(prefix), *(len(prefix) + end for end in ends)))

    def canvas(self, shape: tuple[int, int, int], intra: bool) -> "BatchCanvas":
        return BatchCanvas(self, shape, intra)


class BatchCanvas:
    def __init__(self, bank: BatchRecords, shape: tuple[int, int, int], intra: bool):
        self.bank, self.intra = bank, intra
        self.geometry = bank.geometry(shape)
        self.symbols: NDArray[np.int16] = np.zeros(shape, dtype=np.int16)
        self.records = 0
        self.symbol_count = 0
        self.levels: NDArray[np.uint8] = np.zeros(len(self.geometry.groups), dtype=np.uint8)

    def decode(self, buffer: bytearray) -> tuple[int, int]:
        bands = self.geometry.groups.grid[0] if self.bank.static else 0
        consumed, records, symbols = self.bank.banks[self.intra].receive(
            self.geometry.native, buffer, self.symbols, self.levels, bands, self.bank.raw)
        self.symbol_count += symbols
        self.records += records
        return consumed, records

    def apply_batch(self, records: Sequence[Enhancement]) -> None:
        encoded = bytearray(b"".join(pack_enhancement(record) for record in records))
        consumed, decoded = self.decode(encoded)
        if consumed != len(encoded) or decoded != len(records):
            raise WireError("batch did not contain complete enhancement records")

    def delta(self, step: Tensor) -> Tensor:
        return torch.from_numpy(self.symbols).to(step.device, dtype=torch.float32)[None] * (step * 0.5)


@dataclass(frozen=True)
class Display:
    base: Tensor
    canvas: BatchCanvas
    step: Tensor
    display: Callable[[Tensor], Tensor]


class BatchReceiver:
    def __init__(self, tag: bytes, decode: Callable[[BaseFrame], Display]):
        self.decode = decode
        self.tag = tag
        self.buffer = bytearray()
        self.received = 0
        self.frame: Display | None = None
        self.failed = False
        self.closed = False

    @property
    def base(self) -> Tensor | None:
        return self.frame.base if self.frame is not None else None

    @property
    def canvas(self) -> BatchCanvas | None:
        return self.frame.canvas if self.frame is not None else None

    @property
    def symbol_count(self) -> int:
        return self.frame.canvas.symbol_count if self.frame is not None else 0

    def feed(self, chunk: bytes) -> None:
        self._check()
        if self.closed:
            raise WireError("access unit already closed by transport")
        self.received += len(chunk)
        if self.received > MAX_FRAME_BYTES:
            self.failed = True
            raise WireError("frame exceeds maximum application length")
        self.buffer.extend(chunk)

    def commit_base(self, record: BaseFrame) -> None:
        self._check()
        if self.frame is not None:
            raise WireError("base was already committed")
        self.frame = self.decode(record)
        self.received = max(self.received, APP_BASE_OVERHEAD + len(record.payload))

    def apply_enhancement(self, record: Enhancement) -> None:
        """Apply a parsed record at a complete-record boundary."""
        if self.frame is None:
            raise WireError("enhancement requires a complete base")
        self.finish()
        self.feed(pack_enhancement(record))
        self.decode_available()

    def _receive_base(self) -> int:
        if len(self.buffer) < APP_BASE_OVERHEAD:
            return 0
        size = BASE_HEADER.unpack_from(self.buffer)[-1]
        if not MIN_RANS_BYTES <= size <= MAX_BASE_BYTES:
            raise WireError("invalid base record length")
        end = APP_BASE_OVERHEAD + size
        if len(self.buffer) < end:
            return 0
        parser = FrameParser(self.tag)
        parser.feed(self.buffer[:end])
        [record] = parser.available()
        parser.finish()
        if not isinstance(record, BaseFrame):
            raise WireError("expected a complete base")
        self.commit_base(record)
        del self.buffer[:end]
        return 1

    def decode_available(self) -> int:
        self._check()
        try:
            count = self._receive_base() if self.frame is None else 0
            if self.frame is not None and self.buffer:
                consumed, records = self.frame.canvas.decode(self.buffer)
                del self.buffer[:consumed]
                count += records
            return count
        except ValueError as error:
            self.failed = True
            raise WireError(str(error)) from error

    def _check(self) -> None:
        if self.failed:
            raise WireError("receiver has rejected this frame")

    def finish(self) -> None:
        self._check()
        if self.frame is None or self.buffer:
            raise WireError("incomplete application frame")

    def close(self) -> None:
        """Signal the end of an access unit after all complete records are decoded."""
        self.finish()
        self.closed = True

    def render(self) -> Tensor:
        self._check()
        frame = self.frame
        if frame is None:
            raise WireError("display requires a complete base")
        if frame.canvas.records == 0:
            return frame.base
        return frame.display(frame.canvas.delta(frame.step))


def quantized_symbols(latent: Tensor, step: Tensor) -> NDArray[np.int8]:
    """The historical rounded FP32 alphabet, copied once as signed bytes."""
    scaled = latent.float() / step
    coarse = scaled.round()
    both = torch.stack((coarse, ((scaled - coarse) * 2).round()))
    valid = both.abs() <= 127
    encoded = torch.where(valid, both, -128).to(torch.int8).contiguous().cpu().numpy()[:, 0]
    return encoded
