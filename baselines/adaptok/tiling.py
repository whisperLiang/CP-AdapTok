"""A tile is one fixed spatial position across a complete temporal window."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TileLayout:
    height: int
    width: int
    tile_size: int = 128
    stride: int = 128

    def __post_init__(self):
        if min(self.height, self.width, self.tile_size, self.stride) < 1 or self.stride > self.tile_size:
            raise ValueError("positive dimensions and stride <= tile_size are required")

    def _positions(self, length):
        count = max(1, math.ceil((length - self.tile_size) / self.stride) + 1)
        return tuple(i * self.stride for i in range(count))

    @property
    def coordinates(self):
        return tuple((y, x) for y in self._positions(self.height) for x in self._positions(self.width))

    @property
    def padded_shape(self):
        return self._positions(self.height)[-1] + self.tile_size, self._positions(self.width)[-1] + self.tile_size


def extract_batches(clip, layout, batch_size=4):
    """Yield (coordinates, BCTHW tensor); never crop frames independently."""
    if clip.ndim != 4 or tuple(clip.shape[-2:]) != (layout.height, layout.width) or batch_size < 1:
        raise ValueError("expected TCHW input matching layout and positive batch_size")
    ph, pw = layout.padded_shape
    padded = F.pad(clip, (0, pw - layout.width, 0, ph - layout.height), mode="reflect")
    coordinates, size = layout.coordinates, layout.tile_size
    for start in range(0, len(coordinates), batch_size):
        positions = coordinates[start:start + batch_size]
        tiles = torch.stack([padded[..., y:y + size, x:x + size].permute(1, 0, 2, 3) for y, x in positions])
        yield positions, tiles


def blend_window(size, device):
    # Half-pixel cosine sampling keeps every border strictly positive, including image edges.
    axis = torch.sin(torch.pi * (torch.arange(size, device=device).float() + 0.5) / size).square()
    return axis[:, None] * axis[None, :]


class Stitcher:
    def __init__(self, layout, frames, channels, device):
        self.layout = layout
        ph, pw = layout.padded_shape
        self.output = torch.zeros(frames, channels, ph, pw, device=device)
        self.weights = torch.zeros(ph, pw, device=device)
        self.window = blend_window(layout.tile_size, device) if layout.stride < layout.tile_size else None

    def add(self, coordinates, tiles):
        size = self.layout.tile_size
        for (y, x), tile in zip(coordinates, tiles):
            region = (..., slice(y, y + size), slice(x, x + size))
            video = tile.permute(1, 0, 2, 3).float()
            if self.window is None:
                self.output[region] = video
                self.weights[y:y + size, x:x + size] = 1
            else:
                self.output[region] += video * self.window
                self.weights[y:y + size, x:x + size] += self.window

    def finish(self):
        h, w = self.layout.height, self.layout.width
        weights = self.weights[:h, :w]
        if not bool((weights > 0).all()):
            raise ValueError("uncovered output pixels")
        result = self.output[..., :h, :w]
        return result if self.window is None else result / weights


def stitch(batches, layout):
    first = batches[0][1]
    writer = Stitcher(layout, first.shape[2], first.shape[1], first.device)
    for coordinates, tiles in batches:
        writer.add(coordinates, tiles)
    return writer.finish()
