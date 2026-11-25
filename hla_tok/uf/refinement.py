"""Bounded two-level refinement and dependency-ready scheduling."""

from collections.abc import Iterator
from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, prod

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


class RefinementQuantizer(nn.Module):
    def __init__(self, channels: int, initial_step: float = 0.05):
        super().__init__()
        initial = torch.full((1, channels, 1, 1), (initial_step - 1 / 127) / 0.16)
        self.logit = nn.Parameter(torch.logit(initial))

    def step(self) -> Tensor:
        return 1 / 127 + 0.16 * self.logit.float().sigmoid()

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        step = self.step()
        coarse = _round_ste(latent.float() / step) * step
        fine = _round_ste((latent.float() - coarse.detach()) / (step * 0.5)) * (step * 0.5)
        return coarse, fine

    @torch.no_grad()
    def symbols(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        scaled = latent.float() / self.step()
        coarse = scaled.round()
        fine = ((scaled - coarse) * 2).round()
        symbols = torch.stack((coarse, fine)).cpu()
        if not torch.isfinite(symbols).all() or symbols.abs().max() > 127:
            raise ValueError("refinement exceeds the signed alphabet")
        coarse_i, fine_i = symbols.to(torch.int8).unbind(0)
        return coarse_i, fine_i


def _round_ste(value: Tensor) -> Tensor:
    return value + (value.round() - value).detach()


@dataclass(frozen=True, slots=True)
class GroupLayout:
    shape: tuple[int, int, int]
    tile: int
    channels: int

    @property
    def grid(self) -> tuple[int, int, int]:
        channels, height, width = self.shape
        return ceil(channels / self.channels), ceil(height / self.tile), ceil(width / self.tile)

    def __len__(self) -> int:
        return prod(self.grid)

    def region(self, group: int) -> tuple[slice, slice, slice]:
        if not 0 <= group < len(self):
            raise ValueError("group id outside latent layout")
        bands, _, cols = self.grid
        spatial, band = divmod(group, bands)
        row, col = divmod(spatial, cols)
        origin = (band * self.channels, row * self.tile, col * self.tile)
        extents = (self.channels, self.tile, self.tile)
        channel, height, width = (slice(start, min(start + count, bound))
                                  for start, count, bound in zip(origin, extents, self.shape, strict=True))
        return channel, height, width

    def extent(self, group: int) -> tuple[int, int, int]:
        channel, height, width = self.region(group)
        return channel.stop - channel.start, height.stop - height.start, width.stop - width.start


def priority_order(scores: NDArray[np.float64], active: NDArray[np.bool_]) -> Iterator[tuple[int, int]]:
    queue: list[tuple[float, int, int]] = []
    for active_index in np.flatnonzero(active.any(axis=0)):
        group = int(active_index)
        heappush(queue, (-float(scores[0, group]), 0, group))
    while queue:
        _, level, group = heappop(queue)
        yield level, group
        if level == 0 and active[1, group]:
            heappush(queue, (-float(scores[1, group]), 1, group))
