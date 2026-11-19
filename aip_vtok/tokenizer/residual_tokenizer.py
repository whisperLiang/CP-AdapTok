"""Three genuine quantized residual passes without a global-attention bottleneck."""

from dataclasses import dataclass

import torch
from torch import nn

from aip_vtok.backbones.latent_to_ref import DepthwiseResidual
from aip_vtok.tokenizer.fsq import FSQ
from aip_vtok.tokenizer.halton import halton_order
from aip_vtok.tokenizer.tier_map import ScanPlan, make_tier_map, scan_plan


@dataclass
class TokenEncoding:
    ids: torch.Tensor
    codes: torch.Tensor
    tiers: torch.Tensor
    plan: ScanPlan
    residual1: torch.Tensor
    residual2: torch.Tensor

    def prefix_ids(self, budget):
        if not 0 <= budget <= self.ids.shape[1]:
            raise ValueError("prefix exceeds the single encoded sequence")
        return self.ids[:, :budget]


def strided_block():
    return nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), DepthwiseResidual(128))


def deterministic_lift(layer, codes):
    # GEMM may pick a different reduction when the packet's row count changes.
    # A fixed six-term sum makes each token independent of its batch/packet size.
    value = codes[..., :1] * layer.weight[:, 0]
    for dimension in range(1, 6):
        value = value + codes[..., dimension : dimension + 1] * layer.weight[:, dimension]
    return value + layer.bias


class ResidualTokenizer(nn.Module):
    def __init__(self, maximum=8192):
        super().__init__()
        self.maximum, self.fsq = maximum, FSQ()
        self.anchor = nn.Sequential(strided_block(), strided_block(), nn.Conv2d(128, 6, 1))
        self.innovation = nn.Sequential(strided_block(), nn.Conv2d(128, 6, 1))
        self.detail = nn.Sequential(DepthwiseResidual(128), nn.Conv2d(128, 6, 1))
        self.lifts = nn.ModuleList([nn.Linear(6, 128 * size * size) for size in (4, 2, 1)])
        for layer in self.lifts:
            nn.init.zeros_(layer.bias)

    def lift_tokens(self, codes, positions, pass_id, shape):
        height, width = shape
        size = (4, 2, 1)[pass_id]
        values = deterministic_lift(self.lifts[pass_id], codes).reshape(codes.shape[0], -1, 128, size * size)
        values = values.permute(0, 2, 1, 3).flatten(2)
        base = (positions // (width // size) * size) * width + positions % (width // size) * size
        offsets = torch.arange(size, device=codes.device)[:, None] * width + torch.arange(
            size, device=codes.device
        )
        indices = (base[:, None] + offsets.flatten()[None]).flatten()
        return indices, values

    def synthesize_pass(self, codes, positions, pass_id, shape):
        canvas = codes.new_zeros(codes.shape[0], 128, shape[0] * shape[1])
        if positions.numel():
            indices, values = self.lift_tokens(codes, positions, pass_id, shape)
            canvas = canvas.scatter_add(2, indices[None, None].expand_as(values), values)
        return canvas.reshape(codes.shape[0], 128, *shape)

    def quantize_grid(self, grid, positions):
        values = grid.flatten(2).transpose(1, 2)[:, positions]
        return self.fsq(values)

    def encode(self, latent, prediction):
        shape = latent.shape[-2:]
        positions0 = halton_order(shape[0] // 4, shape[1] // 4, latent.device)
        codes0, ids0 = self.quantize_grid(self.anchor(latent - prediction), positions0)
        reconstructed0 = prediction + self.synthesize_pass(codes0, positions0, 0, shape)
        residual1 = latent - reconstructed0.detach()
        tiers = make_tier_map(residual1)
        plan = scan_plan(tiers, self.maximum).to(latent.device)
        positions1 = plan.positions[plan.pass_ids == 1]
        codes1, ids1 = self.quantize_grid(self.innovation(residual1), positions1)
        reconstructed1 = reconstructed0 + self.synthesize_pass(codes1, positions1, 1, shape)
        residual2 = latent - reconstructed1.detach()
        positions2 = plan.positions[plan.pass_ids == 2]
        codes2, ids2 = self.quantize_grid(self.detail(residual2), positions2)
        return TokenEncoding(
            torch.cat((ids0, ids1, ids2), 1),
            torch.cat((codes0, codes1, codes2), 1),
            tiers,
            plan,
            residual1,
            residual2,
        )
