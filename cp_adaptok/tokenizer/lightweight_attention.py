import math

import torch
from torch import nn
from torch.nn import functional as F


def spatial_positions(height, width, dim, device, dtype):
    """Analytic coordinates for spatial K/V, never an assignment of 1D queries to pixels."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    frequency = torch.arange(1, math.ceil(dim / 4) + 1, device=device) * math.pi
    pos = torch.cat(
        [(c.flatten()[:, None] * frequency).sin() for c in (y, x)]
        + [(c.flatten()[:, None] * frequency).cos() for c in (y, x)],
        dim=-1,
    )
    return pos[:, :dim].to(dtype).unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, spatial):
        b, n, d = query.shape
        q = self.q(self.norm_q(query)).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(self.norm_kv(spatial)).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        query = query + self.out(attended)
        return query + self.ff(self.norm_ff(query))
