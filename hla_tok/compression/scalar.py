"""Train-calibrated fixed scalar quantization independent of sent budget."""

import torch
from torch import Tensor, nn

from hla_tok.compression.packing import scalar_bounds
from hla_tok.compression.transform import _centers, _check_inputs


class StaticScalar(nn.Module):
    def __init__(self, scales: Tensor, bits: int):
        super().__init__()
        self.low, self.high = scalar_bounds(bits)
        if scales.ndim != 2 or not scales.numel() or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("static scales must be finite and positive [rank, dimension]")
        self.bits = bits
        self.register_buffer("scales", scales.float().contiguous().clone())

    def scale(self, values: Tensor, ranks: Tensor) -> Tensor:
        _check_inputs(values, ranks, self.scales.shape[-1])
        return _centers(self.scales, ranks)

    def encode(self, values: Tensor, ranks: Tensor) -> Tensor:
        return (values.float() / self.scale(values, ranks)).round().clamp(self.low, self.high).to(torch.int8)

    def decode(self, codes: Tensor, ranks: Tensor) -> Tensor:
        if codes.dtype != torch.int8:
            raise ValueError("expected signed integer codes")
        return codes.float() * self.scale(codes, ranks)
