"""Static and optionally trainable affine transforms indexed by native Rank."""

import torch
from torch import Tensor, nn

from hla_tok.models.local import TOKEN_DIM


def _check_inputs(values: Tensor, ranks: Tensor, width: int) -> None:
    if values.ndim != 3 or values.shape[-1] != width or values.shape[:2].count(0):
        raise ValueError(f"expected nonempty vectors [batch, count, {width}]")
    if ranks.shape != (values.shape[1],) or ranks.dtype != torch.long or ranks.device != values.device:
        raise ValueError("expected one native int64 Rank ID per transmitted position")


def _multiply(values: Tensor, weights: Tensor, ranks: Tensor) -> Tensor:
    if weights.shape[0] == 1:
        return values @ weights[0]
    matrices = weights.index_select(0, ranks)
    return torch.bmm(values.transpose(0, 1), matrices).transpose(0, 1)


def _centers(center: Tensor, ranks: Tensor) -> Tensor:
    return center if center.shape[0] == 1 else center.index_select(0, ranks)


class CompactProjection(nn.Module):
    def __init__(self, center: Tensor, basis: Tensor):
        super().__init__()
        self.register_buffer("center", center.clone())
        self.weight = nn.Parameter(basis.contiguous().clone(), requires_grad=False)

    def forward(self, values: Tensor, ranks: Tensor) -> Tensor:
        _check_inputs(values, ranks, TOKEN_DIM)
        return _multiply(values - _centers(self.center, ranks), self.weight, ranks)


class CompactLift(nn.Module):
    def __init__(self, center: Tensor, basis: Tensor):
        super().__init__()
        self.bias = nn.Parameter(center.clone(), requires_grad=False)
        self.weight = nn.Parameter(basis.contiguous().clone(), requires_grad=False)

    def forward(self, values: Tensor, ranks: Tensor) -> Tensor:
        _check_inputs(values, ranks, self.weight.shape[1])
        return _multiply(values, self.weight, ranks) + _centers(self.bias, ranks)


class CompactTransform(nn.Module):
    def __init__(self, projection: CompactProjection | None = None, lift: CompactLift | None = None):
        super().__init__()
        if (projection is None) != (lift is None):
            raise ValueError("projection and lift must be installed as a pair")
        self.projection = projection
        self.lift = lift

    @property
    def dimension(self) -> int:
        return TOKEN_DIM if self.projection is None else self.projection.weight.shape[-1]

    @property
    def rank_count(self) -> int:
        return 1 if self.projection is None else self.projection.weight.shape[0]

    @classmethod
    def from_pca(cls, center: Tensor, basis: Tensor) -> "CompactTransform":
        if basis.ndim != 3 or basis.shape[:2] != center.shape or basis.shape[1] != TOKEN_DIM:
            raise ValueError("expected PCA centers [rank, 32] and bases [rank, 32, dimension]")
        return cls(CompactProjection(center, basis), CompactLift(center, basis.transpose(-2, -1)))

    def encode(self, values: Tensor, ranks: Tensor) -> Tensor:
        if self.projection is None:
            _check_inputs(values, ranks, TOKEN_DIM)
            return values
        return self.projection(values, ranks)

    def decode(self, values: Tensor, ranks: Tensor) -> Tensor:
        if self.lift is None:
            _check_inputs(values, ranks, TOKEN_DIM)
            return values
        return self.lift(values, ranks)
