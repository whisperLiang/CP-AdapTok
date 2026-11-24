"""Shared local cross-attention. Each encoder query has its own fixed rank."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hla_tok.progressive.geometry import FEATURE_SIDE, MAX_QUERIES, Geometry, split_regions

DIM = 128
HEADS = 4
TOKEN_DIM = 32
RANK_CHUNK = 32
SPATIAL_QUERIES = FEATURE_SIDE**2
Memory = tuple[Tensor, Tensor]


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(DIM)
        self.memory_norm = nn.LayerNorm(DIM)
        self.query = nn.Linear(DIM, DIM)
        self.key_value = nn.Linear(DIM, 2 * DIM)
        self.output = nn.Linear(DIM, DIM)
        self.ffn = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))

    def prepare(self, context: Tensor) -> Memory:
        batch, length, _ = context.shape
        kv = self.key_value(self.memory_norm(context)).reshape(batch, length, 2, HEADS, DIM // HEADS)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind()
        return key, value

    def forward(self, x: Tensor, memory: Memory) -> Tensor:
        batch, length, _ = x.shape
        q = self.query(self.query_norm(x)).reshape(batch, length, HEADS, DIM // HEADS).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, *memory)
        x = x + self.output(y.transpose(1, 2).reshape(batch, length, DIM))
        return x + self.ffn(x)


class LocalQueryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = nn.Linear(96 * 4, DIM)
        self.fuse = nn.Linear(2 * DIM, DIM)
        self.position = nn.Parameter(torch.randn(SPATIAL_QUERIES, DIM) * 0.02)
        self.query = nn.Parameter(torch.randn(MAX_QUERIES, DIM) * 0.02)
        self.rank = nn.Parameter(torch.randn(MAX_QUERIES, DIM) * 0.02)
        self.layers = nn.ModuleList(CrossBlock() for _ in range(2))
        self.output = nn.Linear(DIM, TOKEN_DIM)

    def prepare(self, fine: Tensor, coarse: Tensor, geometry: Geometry) -> tuple[Memory, ...]:
        fine = split_regions(F.pixel_unshuffle(fine, 2), geometry)
        coarse = split_regions(coarse, geometry)
        features = self.fuse(torch.cat((self.fine(fine), coarse), dim=-1)).flatten(0, 1)
        features = features + self.position.to(features.dtype)
        return tuple(layer.prepare(features) for layer in self.layers)

    def forward(self, memory: tuple[Memory, ...], start: int) -> Tensor:
        # The last block is computed in full even if the caller uses only some rows.
        end = start + RANK_CHUNK
        initial = self.query[start:end] + self.rank[start:end]
        key = memory[0][0]
        x = initial.to(key.dtype).expand(key.shape[0], -1, -1)
        for layer, context in zip(self.layers, memory):
            x = layer(x, context)
        return self.output(x)


class LocalTokenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(TOKEN_DIM, DIM)
        self.rank = nn.Parameter(torch.randn(MAX_QUERIES, DIM) * 0.02)
        self.spatial = nn.Parameter(torch.randn(SPATIAL_QUERIES, DIM) * 0.02)
        self.layers = nn.ModuleList(CrossBlock() for _ in range(2))

    def forward(self, local: Tensor) -> Tensor:
        batch, regions, length, _ = local.shape
        if length == 0:
            return self.spatial.to(local.dtype).expand(batch, regions, -1, -1)
        values = self.input(local.flatten(0, 1))
        context = values + self.rank[:length].to(values.dtype)
        x = self.spatial.to(values.dtype).expand(batch * regions, -1, -1)
        for layer in self.layers:
            x = layer(x, layer.prepare(context))
        return x.unflatten(0, (batch, regions))
