"""Deterministic residual-energy tiers and the sole spatial scan definition."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from aip_vtok.tokenizer.halton import halton_order


@dataclass(frozen=True)
class ScanPlan:
    pass_ids: torch.Tensor
    positions: torch.Tensor
    latent_shape: tuple

    def to(self, device):
        return ScanPlan(self.pass_ids.to(device), self.positions.to(device), self.latent_shape)


def make_tier_map(residual):
    if residual.ndim != 4 or residual.shape[0] != 1:
        raise ValueError("tier map requires a single causal frame")
    energy = F.avg_pool2d(residual.detach().float().square().mean(1, keepdim=True), 2)[0, 0].cpu()
    height, width = energy.shape
    order = halton_order(height, width)
    ranked = order[torch.argsort(energy.flatten()[order], stable=True)]
    count = height * width
    tiers = torch.empty(count, dtype=torch.uint8)
    boundaries = (0, int(count * 0.05), int(count * 0.20), int(count * 0.55), count)
    for tier in range(4):
        tiers[ranked[boundaries[tier] : boundaries[tier + 1]]] = tier
    return tiers.reshape(height, width)


def scan_plan(tiers, maximum=8192):
    if tiers.ndim != 2 or torch.any(tiers > 3):
        raise ValueError("invalid two-bit tier map")
    height, width = tiers.shape
    anchor = halton_order(height // 2, width // 2)
    order = halton_order(height, width)
    flat = tiers.cpu().flatten()
    groups = {tier: order[flat[order] == tier] for tier in (3, 2, 1)}
    pass1 = torch.cat([groups[tier] for tier in (3, 2, 1)])
    parents = torch.cat([groups[tier] for tier in (3, 2)])
    children = halton_order(2, 2)
    base = (parents // width * 2) * (width * 2) + parents % width * 2
    pass2 = (base[:, None] + children // 2 * (width * 2) + children % 2).flatten()
    positions = torch.cat((anchor, pass1, pass2))[:maximum]
    passes = torch.cat((torch.zeros_like(anchor), torch.ones_like(pass1), torch.full_like(pass2, 2)))[
        :maximum
    ]
    return ScanPlan(passes, positions, (height * 2, width * 2))
