"""Mask-aware lightweight latent fusion, followed by exactly one native decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.backbones.latent_to_ref import DepthwiseResidual


class FusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(130, 128, 1), *[DepthwiseResidual(128) for _ in range(3)], nn.Conv2d(128, 128, 1)
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, latent, masks):
        v1 = F.interpolate(masks[1].float(), size=latent.shape[-2:], mode="nearest")
        return latent + self.fusion(torch.cat((latent, v1, masks[2].float()), 1))
