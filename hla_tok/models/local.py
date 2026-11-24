"""Shared local cross-attention. Each encoder query has its own fixed rank."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

from hla_tok.models.anchors import (
    DIRECT_CAPACITY,
    GRID_CELLS,
    anchor_table,
    gaussian_bias,
    position_embedding,
)
from hla_tok.models.rank_causal import RankCausalBlock, RankHistory
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

    def project_query(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return self.query(self.query_norm(x)).reshape(batch, length, HEADS, DIM // HEADS).transpose(1, 2)

    def forward(self, x: Tensor, memory: Memory, bias: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        q = self.project_query(x)
        y = F.scaled_dot_product_attention(q, *memory, attn_mask=bias)
        x = x + self.output(y.transpose(1, 2).reshape(batch, length, DIM))
        return x + self.ffn(x)


class LocalFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = nn.Linear(96 * 4, DIM)
        self.fuse = nn.Linear(2 * DIM, DIM)
        self.position = nn.Parameter(torch.randn(SPATIAL_QUERIES, DIM) * 0.02)

    def features(self, fine: Tensor, coarse: Tensor, geometry: Geometry) -> Tensor:
        fine = split_regions(F.pixel_unshuffle(fine, 2), geometry)
        coarse = split_regions(coarse, geometry)
        features = self.fuse(torch.cat((self.fine(fine), coarse), dim=-1)).flatten(0, 1)
        return features + self.position.to(features.dtype)


class LocalQueryEncoder(LocalFeatureFusion):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(MAX_QUERIES, DIM) * 0.02)
        self.rank = nn.Parameter(torch.randn(MAX_QUERIES, DIM) * 0.02)
        self.layers = nn.ModuleList(CrossBlock() for _ in range(2))
        self.output = nn.Linear(DIM, TOKEN_DIM)

    def prepare(self, fine: Tensor, coarse: Tensor, geometry: Geometry) -> tuple[Memory, ...]:
        features = self.features(fine, coarse, geometry)
        return tuple(layer.prepare(features) for layer in self.layers)

    def initial(self, start: int, key: Tensor) -> Tensor:
        end = start + RANK_CHUNK
        initial = self.query[start:end] + self.rank[start:end]
        return initial.to(key.dtype).expand(key.shape[0], -1, -1)

    def forward(self, memory: tuple[Memory, ...], start: int, history: RankHistory | None = None) -> Tensor:
        # A caller may consume only some positions, while computation uses a full fixed rank chunk.
        x = self.initial(start, memory[0][0])
        for layer, context in zip(self.layers, memory):
            x = layer(x, context)
        return self.output(x)


class DirectProjection(LocalFeatureFusion):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(DIM, TOKEN_DIM) for _ in range(2))
        table = anchor_table()[:DIRECT_CAPACITY]
        self.register_buffer("anchor_table", table)
        self.register_buffer("spatial_indices", table[:, 1] * FEATURE_SIDE + table[:, 0], persistent=False)

    def prepare(self, fine: Tensor, coarse: Tensor, geometry: Geometry) -> Tensor:
        return self.features(fine, coarse, geometry)

    def forward(self, memory: Tensor, start: int, history: RankHistory | None = None) -> Tensor:
        index = self.spatial_indices[start:start + RANK_CHUNK]
        return self.project[start // GRID_CELLS](memory.index_select(1, index))


class AnchoredQueryEncoder(LocalQueryEncoder):
    def __init__(self, causal: bool):
        super().__init__()
        table = anchor_table()
        self.register_buffer("anchor_table", table)
        self.register_buffer("anchor_position", position_embedding(table, DIM), persistent=False)
        self.register_buffer("spatial_bias", gaussian_bias(table), persistent=False)
        self.rank_causal = RankCausalBlock(DIM, HEADS) if causal else None

    def forward(self, memory: tuple[Memory, ...], start: int, history: RankHistory | None = None) -> Tensor:
        ranks = slice(start, start + RANK_CHUNK)
        x = self.initial(start, memory[0][0])
        x = x + self.anchor_position[ranks].to(x.dtype)
        bias = self.spatial_bias[ranks]
        with record_function("hla/anchored_cross_attention"):
            x = self.layers[0](x, memory[0], bias)
        if self.rank_causal is not None:
            if history is None:
                raise ValueError("rank-causal encoder requires frame-owned KV history")
            x = self.rank_causal(x, history, start)
        with record_function("hla/anchored_cross_attention"):
            x = self.layers[1](x, memory[1], bias)
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
