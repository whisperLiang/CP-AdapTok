from dataclasses import dataclass

import torch


@dataclass
class PrefixAttentionCache:
    query: torch.Tensor
    maximum: torch.Tensor
    exp_sum: torch.Tensor
    weighted_value: torch.Tensor
    tokens: int = 0

    @classmethod
    def empty(cls, query):
        shape = (*query.shape[:-1], 1)
        return cls(
            query,
            torch.full(shape, -torch.inf, device=query.device, dtype=query.dtype),
            torch.zeros(shape, device=query.device, dtype=query.dtype),
            torch.zeros_like(query),
        )

    def append(self, key, value, first_index):
        if first_index != self.tokens or key.shape[-2] == 0:
            raise ValueError("cache append must be contiguous and nonempty")
        scores = self.query @ key.to(self.query.dtype).transpose(-2, -1) / self.query.shape[-1] ** 0.5
        block_max = scores.amax(-1, keepdim=True)
        maximum = torch.maximum(self.maximum, block_max)
        previous_scale = torch.exp(self.maximum - maximum)
        exponentials = torch.exp(scores - maximum)
        self.exp_sum = self.exp_sum * previous_scale + exponentials.sum(-1, keepdim=True)
        self.weighted_value = self.weighted_value * previous_scale + exponentials @ value.to(self.query.dtype)
        self.maximum = maximum
        self.tokens += key.shape[-2]
        return self.weighted_value / self.exp_sum
