"""Public frame geometry and the implicit rank-major wire order."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

REGION_SIZE = 128
FEATURE_SIDE = 8
MAX_QUERIES = 512
NATIVE_SHAPE = (1080, 1920)
NATIVE_REGIONS = 135
BUDGETS = (4096, 6144, 8192, 12288, 16320, 24512, 32640, 65280)
BASE_BUDGET = BUDGETS[0]
PRIMARY_BUDGETS = BUDGETS[:5]


@dataclass(frozen=True)
class Geometry:
    height: int
    width: int

    def __post_init__(self):
        dimensions = self.height, self.width
        if dimensions != NATIVE_SHAPE and any(d < REGION_SIZE or d % REGION_SIZE for d in dimensions):
            raise ValueError("expected native 1080p or a crop aligned to 128 pixels in both dimensions")

    @property
    def rows(self) -> int:
        return (self.height + REGION_SIZE - 1) // REGION_SIZE

    @property
    def columns(self) -> int:
        return self.width // REGION_SIZE

    @property
    def num_regions(self) -> int:
        return self.rows * self.columns

    @property
    def capacity(self) -> int:
        return self.num_regions * MAX_QUERIES

    def pad(self, rgb: Tensor) -> Tensor:
        bottom = self.rows * REGION_SIZE - self.height
        return F.pad(rgb, (0, 0, 0, bottom), mode="reflect") if bottom else rgb

    def crop(self, padded: Tensor) -> Tensor:
        return padded[..., :self.height, :self.width]

    def scale_budget(self, native_budget: int) -> int:
        region_lengths(native_budget, NATIVE_REGIONS)
        return max(1, (native_budget * self.num_regions + NATIVE_REGIONS // 2) // NATIVE_REGIONS)

    def check_budget(self, count: int) -> None:
        if type(count) is not int or not 0 < count <= self.capacity:
            raise ValueError(f"expected integer budget in [1, {self.capacity}]")


def split_regions(grid: Tensor, geometry: Geometry) -> Tensor:
    batch, channels, _, _ = grid.shape
    rows, columns, side = geometry.rows, geometry.columns, FEATURE_SIDE
    return (grid.reshape(batch, channels, rows, side, columns, side)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, rows * columns, side * side, channels))


def merge_regions(regions: Tensor, geometry: Geometry) -> Tensor:
    batch, _, _, channels = regions.shape
    rows, columns, side = geometry.rows, geometry.columns, FEATURE_SIDE
    grid = (regions.reshape(batch, rows, columns, side, side, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, rows * side, columns * side, channels))
    return grid.permute(0, 3, 1, 2)


def rank_major(local: Tensor) -> Tensor:
    """Map [batch, region, rank, ...] to [batch, sequence, ...]."""
    return local.transpose(1, 2).flatten(1, 2)


def region_lengths(count: int, regions: int) -> tuple[int, ...]:
    if type(count) is not int or not 0 < count <= regions * MAX_QUERIES:
        raise ValueError("budget exceeds region capacity or is not a positive integer")
    whole, extra = divmod(count, regions)
    return (whole + 1,) * extra + (whole,) * (regions - extra)


def group_prefix(values: Tensor, regions: int) -> tuple[Tensor, ...]:
    """Group received vectors by equal local length, without padding absent ranks."""
    batch, count, dim = values.shape
    whole, extra = divmod(count, regions)
    base = values[:, :whole * regions].reshape(batch, whole, regions, dim).transpose(1, 2)
    if not extra:
        return (base,)
    longer = torch.cat((base[:, :extra], values[:, -extra:, None]), dim=2)
    return longer, base[:, extra:]
