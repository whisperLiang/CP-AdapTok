"""Synchronized previous reference latent -> current prediction, strictly causal."""

import torch
from torch import nn

from aip_vtok.backbones.latent_to_ref import DepthwiseResidual


class TemporalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[DepthwiseResidual(128) for _ in range(3)])
        for block in self.blocks:
            nn.init.zeros_(block.net[-1].weight)
            nn.init.zeros_(block.net[-1].bias)

    def forward(self, previous, intra, shape, device):
        if intra:
            return torch.zeros(1, 128, *shape, device=device)
        if previous is None:
            raise ValueError("P prediction requires a committed reference latent")
        return self.blocks(previous)
