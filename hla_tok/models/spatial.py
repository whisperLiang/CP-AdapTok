"""Convolutional analysis and synthesis at native pixel scale."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

from hla_tok.progressive.geometry import Geometry


class LiteBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.project = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.depthwise(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.project(F.gelu(self.expand(y)))


def blocks(channels: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*(LiteBlock(channels) for _ in range(depth)))


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.PixelUnshuffle(4), nn.Conv2d(48, 48, 1), blocks(48, 2))
        self.down_one = nn.Sequential(nn.Conv2d(48, 96, 3, stride=2, padding=1), blocks(96, 2))
        self.down_two = nn.Sequential(nn.Conv2d(96, 128, 3, stride=2, padding=1), blocks(128, 3))

    def forward(self, rgb: Tensor, geometry: Geometry) -> tuple[Tensor, Tensor]:
        stem = self.stem(geometry.pad(rgb).contiguous(memory_format=torch.channels_last))
        fine = self.down_one(stem)
        return fine, self.down_two(fine)


class RGBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = blocks(128, 3)
        self.upsample = nn.Sequential(
            nn.Conv2d(128, 96 * 4, 3, padding=1), nn.PixelShuffle(2), blocks(96, 2),
            nn.Conv2d(96, 48 * 4, 3, padding=1), nn.PixelShuffle(2), blocks(48, 2),
            nn.Conv2d(48, 3 * 16, 3, padding=1), nn.PixelShuffle(4),
        )

    def forward(self, grid: Tensor, geometry: Geometry) -> Tensor:
        with record_function("hla/cross_region_fusion"):
            fused = self.fusion(grid)
        with record_function("hla/rgb_decoder"):
            return geometry.crop(self.upsample(fused)).float().sigmoid()
