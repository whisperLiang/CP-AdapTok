"""Six finite scalar quantizers with an exact mixed-radix 64,000-ID bijection.

Codes are normalized integer centers. Even levels use integers [-4,3], with the
half-bin offset inside tanh; nine centers [-4,4] would not be an 8-level FSQ.
"""

import torch
from torch import nn


class FSQ(nn.Module):
    def __init__(self):
        super().__init__()
        levels = torch.tensor([8, 8, 8, 5, 5, 5], dtype=torch.long)
        self.register_buffer("levels", levels)
        self.register_buffer("radix", torch.cat((torch.ones(1, dtype=torch.long), levels.cumprod(0)[:-1])))
        self.register_buffer("half_width", levels // 2)

    def forward(self, x):
        x = x.float()  # FP32 scalar bin boundaries under training autocast.
        extent = (self.levels.to(x.dtype) - 1) / 2
        offset = (self.levels % 2 == 0).to(x.dtype) / 2
        shift = torch.atanh(offset / extent)
        bounded = torch.tanh(x + shift) * extent - offset
        quantized = bounded + (bounded.round() - bounded).detach()
        codes = quantized / self.half_width
        return codes, self.codes_to_ids(codes.detach())

    def codes_to_ids(self, codes):
        digits = (codes * self.half_width).round().long() + self.half_width
        return self.digits_to_ids(digits)

    def digits_to_ids(self, digits):
        if digits.shape[-1] != 6 or torch.any(digits < 0) or torch.any(digits >= self.levels):
            raise ValueError("invalid FSQ digits")
        return (digits * self.radix).sum(-1)

    def ids_to_digits(self, ids):
        if ids.is_floating_point() or torch.any(ids < 0) or torch.any(ids >= 64000):
            raise ValueError("FSQ ID outside [0,63999]")
        return ids[..., None] // self.radix % self.levels

    def ids_to_codes(self, ids):
        return (self.ids_to_digits(ids) - self.half_width).float() / self.half_width
