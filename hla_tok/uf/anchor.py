"""Frozen official base codec plus a bounded RGB residual analysis/refinement head."""

import hashlib
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from math import prod

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from hla_tok.uf.base import BaseFrame
from hla_tok.uf.entropy import MULTIPLIERS, EntropyTables, SegmentCoder
from hla_tok.uf.refinement import GroupLayout, RefinementQuantizer, priority_order
from hla_tok.uf.wire import RECORD_OVERHEAD, Enhancement, FrameParser, WireError

STRIDE = 8
MAX_APPLICATION_BYTES = 166666
BUDGETS = {5: 20833, 10: 41666, 20: 83333, 40: MAX_APPLICATION_BYTES}


class AnchorModel(nn.Module):
    channels = 12

    def __init__(self) -> None:
        super().__init__()
        planes = 3 * STRIDE**2
        self.analysis = nn.Conv2d(planes, self.channels, 1, bias=False)
        self.local = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels, bias=False)
        self.refine = nn.Conv2d(self.channels, planes, 1, bias=False)
        self.quantizer = RefinementQuantizer(self.channels)
        self.log_scale = nn.Parameter(torch.tensor([0.7, -0.8]))
        self._initialize()

    @torch.no_grad()
    def _initialize(self) -> None:
        leading = self.analysis.weight.view(self.channels, 3, STRIDE, STRIDE)
        leading.zero_()
        self.local.weight.zero_()
        self.refine.weight.zero_()
        half = STRIDE // 2
        for index in range(self.channels):
            rgb, quadrant = divmod(index, 4)
            row, col = divmod(quadrant, 2)
            leading[index, rgb, row * half:(row + 1) * half, col * half:(col + 1) * half] = 1 / half**2
        self.refine.weight.copy_(self.analysis.weight.transpose(1, 0) * half**2)

    def analyze(self, rgb: Tensor, base: Tensor) -> Tensor:
        lifted = self.analysis(F.pixel_unshuffle(rgb - base, STRIDE))
        return (lifted + self.local(lifted)).float().tanh()

    def display(self, base: Tensor, delta: Tensor) -> Tensor:
        return (base + F.pixel_shuffle(self.refine(delta), STRIDE)).clamp(0, 1)

    def scales(self) -> Tensor:
        return self.log_scale.float().clamp(-3, 3).exp()


@dataclass(frozen=True, slots=True)
class Segmentation:
    tile: int = 16
    channels: int = 12
    conditional: bool = True
    ranked: bool = True


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


def stream_tag(model: nn.Module, identifier: str) -> bytes:
    digest = hashlib.sha256(identifier.encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.digest()[:8]


class RefinementCodec:
    def __init__(self, model: AnchorModel, precision: str, segmentation: Segmentation = Segmentation()):
        if precision not in ("bf16", "fp32"):
            raise ValueError("enhancement precision must be bf16 or fp32")
        self.model, self.precision = model, precision
        self.records_codec = RecordCodec(tuple(model.scales().detach().cpu().tolist()), segmentation)
        tag = f"anchor-v1/{precision}/{segmentation}/{self.records_codec.tables.cdf}"
        self.tag = stream_tag(model, tag)
        self.step = model.quantizer.step().detach()

    def compute(self, rgb: Tensor):
        return torch.autocast("cuda", dtype=torch.bfloat16) if rgb.is_cuda and self.precision == "bf16" else nullcontext()

    def records(self, latent: Tensor) -> Iterator[Enhancement]:
        return self.records_codec.records(*self.model.quantizer.symbols(latent))

    def canvas(self, base: Tensor) -> IntegerCanvas:
        height, width = base.shape[-2:]
        if height % STRIDE or width % STRIDE:
            raise ValueError("anchor requires RGB dimensions divisible by eight")
        return IntegerCanvas(self.records_codec, (self.model.channels, height // STRIDE, width // STRIDE))

    def prepare(self, base: Tensor) -> "DisplayFrame":
        def render(delta: Tensor) -> Tensor:
            with self.compute(base):
                return self.model.display(base, delta)

        return DisplayFrame(base, self.canvas(base), self.step, render)


@dataclass(frozen=True)
class DisplayFrame:
    base: Tensor
    canvas: IntegerCanvas
    step: Tensor
    display: Callable[[Tensor], Tensor]


class FrameReceiver:
    def __init__(self, tag: bytes, decode: Callable[[BaseFrame], DisplayFrame]):
        self.decode = decode
        self.parser = FrameParser(tag)
        self.frame: DisplayFrame | None = None

    @property
    def base(self) -> Tensor | None:
        return self.frame.base if self.frame is not None else None

    @property
    def canvas(self) -> IntegerCanvas | None:
        return self.frame.canvas if self.frame is not None else None

    @property
    def symbol_count(self) -> int:
        return self.frame.canvas.symbol_count if self.frame is not None else 0

    def feed(self, chunk: bytes) -> None:
        self.parser.feed(chunk)

    def decode_available(self) -> int:
        count = 0
        for record in self.parser.available():
            if isinstance(record, BaseFrame):
                self.commit_base(record)
            elif self.frame is not None:
                self.frame.canvas.apply(record)
            else:
                raise WireError("missing committed base")
            count += 1
        return count

    def commit_base(self, record: BaseFrame) -> None:
        if self.frame is not None:
            raise WireError("base was already committed")
        self.frame = self.decode(record)

    def finish(self) -> None:
        self.parser.finish()

    def render(self) -> Tensor:
        frame = self.frame
        if frame is None:
            raise WireError("display requires a complete base")
        if frame.canvas.records == 0:
            return frame.base
        return frame.display(frame.canvas.delta(frame.step))


class ProgressiveReceiver(FrameReceiver):
    def __init__(self, codec: RefinementCodec, decode: Callable[[BaseFrame], Tensor]):
        super().__init__(codec.tag, lambda base: codec.prepare(decode(base)))


def decode_prefix(codec: RefinementCodec, decode: Callable[[BaseFrame], Tensor], raw: bytes) -> ProgressiveReceiver:
    receiver = ProgressiveReceiver(codec, decode)
    receiver.feed(raw)
    receiver.decode_available()
    receiver.finish()
    return receiver
