"""Axis statistics and rank truncation in aligned or physical coordinates."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from aip_vtok.diagnostics.diag0.sparse import fit_scale

from .scan import GROUP_WIDTH, TOP

RANKS = (6, 12, 18, 24, 32, 48, 128)
FILLS = ("zero_fill", "train_mean_fill")
ALIGNED = "aligned"
NATIVE = "p_native"
CHANNELS = 128
PADDED = 132


def native_gradient(grad_aligned, alignment):
    """If A = (Y - mu) / sigma, dRGB_loss/dY = dRGB_loss/dA / sigma."""
    return grad_aligned / alignment.p_std


def coordinate_value(native, aligned, space):
    return aligned if space == ALIGNED else native


def aligned_value(backbone, value, space):
    return value if space == ALIGNED else backbone.alignment.align(value, False)


@dataclass(frozen=True)
class ChannelFit:
    mean: torch.Tensor
    std: torch.Tensor
    energy: torch.Tensor
    scales: torch.Tensor

    @property
    def indices(self):
        return torch.argsort(self.energy, descending=True, stable=True)

    @property
    def weights(self):
        """Positive bounded weights for the normalized latent anchor."""
        relative = self.energy / self.energy.mean().clamp_min(1e-30)
        weights = relative.clamp(0.1, 10)
        return weights / weights.mean()

    def group_energy(self):
        energy = self.energy[self.indices[:TOP]].reshape(-1, GROUP_WIDTH).sum(1)
        return energy / energy.sum().clamp_min(1e-30)

    def order_values(self, latent):
        return latent.index_select(1, self.indices)

    def restore_rank(self, values, fill="zero_fill"):
        if fill not in FILLS:
            raise ValueError(f"unknown channel fill {fill}")
        initial = torch.zeros_like(self.mean) if fill == "zero_fill" else self.mean
        base = initial.view(1, -1, 1, 1).expand(values.shape[0], -1, *values.shape[-2:])
        return base.index_copy(1, self.indices[:values.shape[1]], values)

    def record(self):
        return {name: value.tolist() for name, value in vars(self).items()} | {"order": self.indices.tolist()}


def fit_channels(rows, energy):
    """Rows are fixed-stride Train samples with shape [N, 128]."""
    samples = rows.double()
    std, mean = torch.std_mean(samples, dim=0, correction=0)
    sorted_rows = rows.index_select(1, energy.argsort(descending=True, stable=True))
    padded = F.pad(sorted_rows, (0, PADDED - CHANNELS))
    scales = fit_scale(padded).reshape(-1, GROUP_WIDTH)
    return ChannelFit(mean.float(), std.float().clamp_min(1e-5), energy, scales)
