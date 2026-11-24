"""A local rank-causal self-attention block with absolute-index KV history."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function


def rank_causal_mask(start: int, length: int, device=None) -> Tensor:
    queries = torch.arange(start, start + length, device=device)
    keys = torch.arange(start + length, device=device)
    return keys[None, :] <= queries[:, None]


@dataclass
class RankHistory:
    key: Tensor | None = None
    value: Tensor | None = None

    def append(self, key: Tensor, value: Tensor, start: int) -> tuple[Tensor, Tensor]:
        past = 0 if self.key is None else self.key.shape[-2]
        if start != past:
            raise ValueError(f"rank cache spans {past} previous ranks, requested start {start}")
        if self.key is not None:
            key = torch.cat((self.key, key), dim=-2)
            value = torch.cat((self.value, value), dim=-2)
        self.key, self.value = key, value
        return key, value

    @property
    def nbytes(self) -> int:
        return sum(item.numel() * item.element_size() for item in (self.key, self.value) if item is not None)


class RankCausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, values: Tensor, history: RankHistory, start: int) -> Tensor:
        batch, length, dim = values.shape
        with record_function("hla/rank_causal_attention"):
            projected = self.qkv(self.norm(values)).reshape(batch, length, 3, self.heads, dim // self.heads)
            q, k, v = projected.permute(2, 0, 3, 1, 4).unbind()
            past_k, past_v = history.append(k, v, start)
            allowed = rank_causal_mask(start, length, values.device)
            mixed = F.scaled_dot_product_attention(q, past_k, past_v, attn_mask=allowed)
            values = values + self.projection(mixed.transpose(1, 2).reshape(batch, length, dim))
            return values + self.ffn(values)
