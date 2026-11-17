"""Actual window partitioning: attention length is window**2, including edges."""

import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    def __init__(self, channels, heads=6, window=4):
        super().__init__()
        self.heads, self.window = heads, window
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.project = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        size = self.window
        ph, pw = (-h) % size, (-w) % size
        padded = F.pad(x, (0, pw, 0, ph))
        nh, nw = (h + ph) // size, (w + pw) // size
        windows = padded.reshape(b, c, nh, size, nw, size).permute(0, 2, 4, 3, 5, 1)
        windows = windows.reshape(b * nh * nw, size * size, c)
        valid = torch.ones(b, 1, h, w, device=x.device, dtype=torch.bool)
        valid = F.pad(valid, (0, pw, 0, ph), value=False)
        valid = valid.reshape(b, 1, nh, size, nw, size).permute(0, 2, 4, 3, 5, 1)
        valid = valid.reshape(b * nh * nw, 1, 1, size * size)
        qkv = self.qkv(self.norm(windows)).reshape(-1, size * size, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=valid, dropout_p=0.0)
        mixed = mixed.transpose(1, 2).reshape(-1, size * size, c)
        result = windows + self.project(mixed)
        result = result.reshape(b, nh, nw, size, size, c).permute(0, 5, 1, 3, 2, 4)
        return result.reshape(b, c, h + ph, w + pw)[:, :, :h, :w].contiguous()
