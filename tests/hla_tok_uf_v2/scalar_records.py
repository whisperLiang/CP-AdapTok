"""Scalar encoder and accumulator used to verify the batched D1 wire format."""

from collections.abc import Iterator
from math import prod

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from baselines.hla_tok_uf_v2.entropy import MULTIPLIERS, EntropyTables, SegmentCoder
from baselines.hla_tok_uf_v2.latent import Segmentation
from baselines.hla_tok_uf_v2.refinement import GroupLayout, priority_order
from baselines.hla_tok_uf_v2.wire import RECORD_OVERHEAD, Enhancement, WireError


class RecordCodec:
    def __init__(self, scales: tuple[float, ...], segmentation: Segmentation):
        self.scales = scales
        self.segmentation = segmentation
        self.tables = EntropyTables.from_scales(scales)
        self.coder = SegmentCoder(self.tables)

    def layout(self, shape: tuple[int, int, int]) -> GroupLayout:
        return GroupLayout(shape, self.segmentation.tile, self.segmentation.channels)

    def records(self, coarse: Tensor, fine: Tensor) -> Iterator[Enhancement]:
        array = (coarse.numpy()[0], fine.numpy()[0])
        channels, height, width = coarse.shape[1:]
        layout = self.layout((channels, height, width))
        regions = [layout.region(group) for group in range(len(layout))]
        scores, active, cdfs = self._scores(array, layout)
        order = priority_order(scores, active) if self.segmentation.ranked else _scan(active)
        for level, group in order:
            cdf = int(cdfs[level, group])
            vector = array[level][regions[group]].ravel()
            yield Enhancement(group, level, cdf, self.coder.encode(vector, cdf))

    def _scores(self, symbols: tuple[NDArray[np.int8], NDArray[np.int8]], layout: GroupLayout):
        bands, rows, cols = layout.grid
        channels, tile = layout.channels, layout.tile
        bounds = (bands * channels, rows * tile, cols * tile)
        remaining = tuple((0, upper - lower) for upper, lower in zip(bounds, layout.shape, strict=True))
        padded = np.pad(np.stack(symbols), ((0, 0), *remaining))
        groups = padded.reshape(2, bands, channels, rows, tile, cols, tile).transpose(0, 3, 5, 1, 2, 4, 6)
        counts = np.fromiter((prod(layout.extent(group)) for group in range(len(layout))), dtype=np.int32)
        magnitude = np.abs(groups).sum(axis=(4, 5, 6)).reshape(2, -1) / counts
        scale_levels = np.asarray(self.scales)[:, None]
        buckets = np.ones_like(magnitude, dtype=np.int32)
        if self.segmentation.conditional:
            buckets = (np.rint(np.log2(np.maximum(0.125, magnitude) / scale_levels)) + 1).clip(0, 3).astype(np.int32)
        scale = scale_levels * np.asarray(MULTIPLIERS)[buckets]
        bits = np.maximum(0.08, np.log2(2 * scale)) + magnitude / (scale * np.log(2))
        cost = RECORD_OVERHEAD + 4 + counts * bits / 8
        gain = np.square(groups, dtype=np.int16).sum(axis=(4, 5, 6)).reshape(2, -1) / np.array([[1], [4]])
        return gain / cost, gain != 0, buckets + np.array([[0], [len(MULTIPLIERS)]])

def _scan(active: NDArray[np.bool_]) -> Iterator[tuple[int, int]]:
    for group in np.flatnonzero(active.any(axis=0)):
        yield 0, int(group)
    for group in np.flatnonzero(active[1]):
        yield 1, int(group)

class IntegerCanvas:
    def __init__(self, codec: RecordCodec, shape: tuple[int, int, int]):
        self.codec, self.layout = codec, codec.layout(shape)
        self.symbols = np.zeros(shape, dtype=np.int16)
        self.records = 0
        self.symbol_count = 0

    def apply(self, record: Enhancement) -> None:
        if record.cdf // len(MULTIPLIERS) != record.level:
            raise WireError("CDF does not belong to refinement level")
        extent = self.layout.extent(record.group)
        count = prod(extent)
        values = self.codec.coder.decode(record.payload, record.cdf, count).reshape(extent)
        self.symbols[self.layout.region(record.group)] += values.astype(np.int16) * (2 >> record.level)
        self.records += 1
        self.symbol_count += count

    def delta(self, step: Tensor) -> Tensor:
        return torch.from_numpy(self.symbols).to(step.device, dtype=torch.float32)[None] * (step * 0.5)
