import torch
from torch import nn

from .prefix_cache import PrefixAttentionCache


class TokenGridProjector(nn.Module):
    """A single fixed query cross-attention; all K/V projections are token independent."""

    def __init__(self, dim=192, heads=6, n_max=2048, grid=(17, 30)):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.grid, self.n_max = dim, heads, tuple(grid), n_max
        self.query = nn.Parameter(torch.randn(1, grid[0] * grid[1], dim) * .02)
        self.positions = nn.Parameter(torch.randn(1, n_max, dim) * .02)
        self.token_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose(1, 2)

    def _keys_values(self, tokens, first=0):
        end = first + tokens.shape[1]
        if end > self.n_max:
            raise ValueError("prefix exceeds model N_max")
        k, v = self.kv(self.token_norm(tokens + self.positions[:, first:end])).chunk(2, dim=-1)
        return self._heads(k), self._heads(v)

    def _finish(self, attention):
        b = attention.shape[0]
        attended = attention.transpose(1, 2).reshape(b, -1, self.dim)
        grid = self.query + self.out(attended.to(self.query.dtype))
        grid = grid + self.ff(self.norm(grid))
        return grid.transpose(1, 2).reshape(b, self.dim, *self.grid)

    def forward(self, tokens, mask=None):
        if tokens.shape[1] == 0:
            raise ValueError("empty prefix")
        q = self._heads(self.q(self.query.expand(tokens.shape[0], -1, -1)))
        k, v = self._keys_values(tokens)
        scores = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** .5
        if mask is not None:
            if mask.shape != tokens.shape[:2] or not mask.any(-1).all():
                raise ValueError("mask must include at least one token per sample")
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), -torch.inf)
        attention = scores.softmax(-1) @ v.float()
        return self._finish(attention)

    def new_cache(self, batch_size):
        q = self._heads(self.q(self.query.expand(batch_size, -1, -1)))
        return PrefixAttentionCache.empty(q)

    def append(self, cache, tokens, first_index):
        if self.training:
            raise RuntimeError("prefix cache is inference only")
        k, v = self._keys_values(tokens, first_index)
        return self._finish(cache.append(k, v, first_index))
