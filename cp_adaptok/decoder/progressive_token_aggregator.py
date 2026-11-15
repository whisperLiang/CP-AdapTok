"""Shared packet attention with a prefix-independent, bounded additive scale."""

import math

import torch
from torch import nn

from .prefix_cache import ProgressiveTokenCache
from .token_grid_projector import TokenGridProjector


class SharedProgressiveTokenAggregator(TokenGridProjector):
    """Only aggregation changes; grid queries, FFN and latent decoder stay shared.

    Each group's softmax sees exactly one received packet. Increments are accumulated
    in ascending packet order in the configured precision. Dense and cached decoding
    use that same order, rather than relying on associative floating-point addition.
    """

    def __init__(self, dim=192, heads=6, n_max=2048, grid=(17, 30), precision="float64", packet_tokens=128):
        super().__init__(dim, heads, n_max, grid, precision)
        if packet_tokens < 1 or n_max % packet_tokens:
            raise ValueError("N_max must contain complete packets")
        self.packet_tokens = packet_tokens
        self.group_scale = 1 / math.sqrt(n_max // packet_tokens)
        self.gate = nn.Linear(dim + 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _validate_tokens(self, tokens, first=0):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("expected [B,K,dim] packet embeddings")
        if first < 0 or first % self.packet_tokens or not tokens.shape[1] or tokens.shape[1] % self.packet_tokens:
            raise ValueError("prefix and append must use complete contiguous packets")
        if first + tokens.shape[1] > self.n_max:
            raise ValueError("prefix exceeds model N_max")

    def _validate_mask(self, tokens, mask):
        if mask is None:
            return
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean packet prefix")
        lengths = mask.sum(-1)
        expected = torch.arange(tokens.shape[1], device=mask.device)[None] < lengths[:, None]
        if (lengths < 1).any() or (lengths % self.packet_tokens).any() or not torch.equal(mask, expected):
            raise ValueError("mask must contain complete contiguous packet prefixes")

    def group_gate(self, tokens, first, include_positions=True):
        positioned = tokens + self.positions[:, first:first + tokens.shape[1]] if include_positions else tokens
        content = self.token_norm(positioned).mean(1)
        order = content.new_full((len(content), 1), first / self.n_max)
        # Bounded by 2/sqrt(max_groups); no normalization over the currently received K.
        return 2 * self.group_scale * self.gate(torch.cat((content, order), -1)).sigmoid()

    def _group_delta(self, query, tokens, first, include_positions=True):
        key, value = self._keys_values(tokens, first, include_positions)
        scores = query @ key.to(query.dtype).transpose(-1, -2) / math.sqrt(query.shape[-1])
        attended = scores.softmax(-1) @ value.to(query.dtype)
        gate = self.group_gate(tokens, first, include_positions).to(query.dtype)[:, :, None, None]
        return gate * attended

    def forward(self, tokens, mask=None):
        self._validate_tokens(tokens)
        self._validate_mask(tokens, mask)
        cache = self.new_cache(tokens.shape[0])
        for first in range(0, tokens.shape[1], self.packet_tokens):
            delta = self._group_delta(cache.query, tokens[:, first:first + self.packet_tokens], first)
            if mask is not None:
                delta = delta * mask[:, first, None, None, None]
            cache.accumulated = cache.accumulated + delta
        return self._finish(cache.accumulated)

    def new_cache(self, batch_size):
        query = self._heads(self.q(self.query.expand(batch_size, -1, -1))).to(self.accumulation_dtype)
        return ProgressiveTokenCache(query, torch.zeros_like(query))

    def append(self, cache, tokens, first_index):
        if self.training:
            raise RuntimeError("prefix cache is inference only")
        self._validate_tokens(tokens, first_index)
        if first_index != cache.tokens or tokens.shape[0] != cache.query.shape[0]:
            raise ValueError("cache append must be contiguous and keep its batch size")
        for offset in range(0, tokens.shape[1], self.packet_tokens):
            delta = self._group_delta(cache.query, tokens[:, offset:offset + self.packet_tokens], first_index + offset)
            cache.accumulated = cache.accumulated + delta
            cache.tokens += self.packet_tokens
        return self.decode_grid(cache)

    def decode_grid(self, cache):
        if cache.tokens == 0:
            raise ValueError("cannot decode an empty cache")
        return self._finish(cache.accumulated)
