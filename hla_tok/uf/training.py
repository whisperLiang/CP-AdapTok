"""Train-only multirate STE objective for the frozen-base Anchor."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hla_tok.uf.anchor import BUDGETS, AnchorModel, Segmentation
from hla_tok.uf.entropy import MULTIPLIERS
from hla_tok.uf.refinement import priority_order
from hla_tok.uf.wire import RECORD_OVERHEAD

TILE = Segmentation().tile
WEIGHTS = (0.35, 0.30, 0.20, 0.15)
LAMBDA = (0.00035, 0.00028, 0.00018, 0.0001)
LPIPS_WEIGHT = 0.03
MONOTONIC_WEIGHT = 0.10
BASE_WEIGHT = 0.10


def _group_sum(values: Tensor) -> Tensor:
    levels, batch, channels, rows, cols = values.shape
    if rows % TILE or cols % TILE:
        raise ValueError("training crop must align to complete token groups")
    pooled = F.avg_pool2d(values.flatten(0, 1).sum(dim=1, keepdim=True), TILE) * TILE**2
    return pooled.view(levels, batch, rows // TILE, cols // TILE)


def _record_estimate(symbols: Tensor, scales: Tensor) -> Tensor:
    group_abs = _group_sum(symbols.abs().detach()) / (symbols.shape[2] * TILE**2)
    scale_levels = scales[:, None, None, None]
    chosen = ((group_abs.clamp_min(0.125) / scale_levels.detach()).log2().round() + 1).clamp(0, 3)
    factors = torch.as_tensor(MULTIPLIERS, device=symbols.device)[chosen.long()]
    scale = (scale_levels * factors).repeat_interleave(TILE, -1).repeat_interleave(TILE, -2).unsqueeze(2)
    zero = -torch.expm1(-0.5 / scale)
    tails = (-torch.expm1(-1 / scale) * 0.5) * torch.exp(-(symbols.abs() - 0.5).clamp_min(0) / scale)
    probability = torch.where(symbols.detach().abs() < 0.5, zero, tails)
    return _group_sum(-probability.clamp_min(2**-16).log2()) + (4 + RECORD_OVERHEAD) * 8


def _choose(gains: Tensor, costs: Tensor, base_bytes: Sequence[int], ratio: float) -> Tensor:
    levels, batch, rows, cols = costs.shape
    predicted = costs.detach().reshape(levels, batch, -1).cpu().numpy()
    gain = gains.detach().reshape(levels, batch, -1).cpu().numpy()
    flags = np.zeros((len(BUDGETS), levels, batch, rows * cols), dtype=bool)
    for index, base in enumerate(base_bytes):
        limits = (np.asarray(tuple(BUDGETS.values())) - base).clip(min=0) * ratio * 8
        accrued = 0.0
        for level, group in priority_order(gain[:, index] / predicted[:, index], gain[:, index] > 0):
            accrued += float(predicted[level, index, group])
            if accrued > limits[-1]:
                break
            flags[accrued <= limits, level, index, group] = True
    return torch.from_numpy(flags.reshape(len(BUDGETS), levels, batch, rows, cols)).to(costs.device)


def prefix_objective(model: AnchorModel, rgb: Tensor, base: Tensor, base_bytes: Sequence[int],
                     lpips: nn.Module | None) -> tuple[Tensor, dict[str, Tensor]]:
    pixels = rgb.shape[-2] * rgb.shape[-1]
    if len(base_bytes) != rgb.shape[0]:
        raise ValueError("one real base byte count per training frame is required")
    ratio = pixels / (1920 * 1080)
    latent = model.analyze(rgb, base)
    coarse, fine = model.quantizer(latent)
    deltas = torch.stack((coarse, fine))
    steps = torch.stack((model.quantizer.step(), model.quantizer.step() * 0.5))
    costs = _record_estimate(deltas / steps, model.scales())
    mask = _choose(_group_sum(deltas.square()), costs, base_bytes, ratio)
    render_mask = mask.repeat_interleave(TILE, -1).repeat_interleave(TILE, -2).unsqueeze(3)
    transmitted = (render_mask * deltas.unsqueeze(0)).sum(dim=1)
    total = rgb.new_tensor(base_bytes) * (ratio * 8) + (mask * costs.unsqueeze(0)).sum(dim=(1, 3, 4))
    bpp = total / pixels
    render = model.display(base.repeat(len(BUDGETS), 1, 1, 1), transmitted.flatten(0, 1))
    target = rgb.repeat(len(BUDGETS), 1, 1, 1)
    distortions = (render.float() - target).square().mean(dim=(1, 2, 3))
    if lpips is not None:
        with torch.autocast(rgb.device.type, enabled=False):
            distances = lpips(target.float() * 2 - 1, render.float() * 2 - 1)
        distortions = distortions + LPIPS_WEIGHT * distances.flatten()
    distortion = distortions.view(len(BUDGETS), rgb.shape[0])
    mono = (distortion[1:] - distortion[:-1]).relu().mean()
    base_rd = F.mse_loss(base.float(), rgb.float()) + LAMBDA[0] * rgb.new_tensor(base_bytes).mean() * 8 / (1920 * 1080)
    rd = distortion.mean(dim=1) + rgb.new_tensor(LAMBDA) * bpp.mean(dim=1)
    loss = torch.dot(rgb.new_tensor(WEIGHTS), rd) + MONOTONIC_WEIGHT * mono + BASE_WEIGHT * base_rd
    return loss, {"estimated_bpp": bpp.detach(), "distortion": distortion.detach(), "base_rd": base_rd.detach()}
