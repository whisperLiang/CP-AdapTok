"""Training-only spatial targets for the incremental, transmitted Fine embeddings."""

import torch
from torch.nn import functional as F


def ordered_bins(plane, grid, indices):
    height, width = grid
    pad = (0, width * 32 - plane.shape[-1], 0, height * 32 - plane.shape[-2])
    sums = F.avg_pool2d(F.pad(plane, pad), 8, 8, divisor_override=1)
    tiles = sums.reshape(-1, height, 4, width, 4).permute(0, 1, 3, 2, 4)
    return tiles.reshape(-1, height * width, 16)[:, indices]


@torch.no_grad()
def detail_targets(rgb, coarse, grid, indices):
    diff = rgb.float() - coarse.float()
    luma = .2126 * diff[:, :1] + .7152 * diff[:, 1:2] + .0722 * diff[:, 2:]
    sums = ordered_bins(luma, grid, indices)
    counts = ordered_bins(torch.ones_like(luma[:1]), grid, indices)
    valid = counts > 0
    center = sums.sum(-1, keepdim=True) / counts.sum(-1, keepdim=True).clamp_min(1)
    return (sums / counts.clamp_min(1) - center) * valid, valid


def suffix_angular_loss(values, target, valid):
    direction = F.normalize(values.float() * valid, dim=-1, eps=1e-6)
    # RMS(t) * (1 - cos(q, t)), including the exact zero-energy limit.
    scale = valid.sum(-1).clamp_min(1).sqrt()
    return ((target.norm(dim=-1) - (direction * target).sum(-1)) / scale).mean()


def suffix_supervision(model, frame, state, encoded, outputs, weight):
    budgets = model.config.budgets
    if not weight or budgets[-1] not in outputs:
        return frame.new_zeros(())
    low, high = budgets[-2:]
    offset = sum(model.positions.counts[:2])
    if low < offset:
        raise ValueError("suffix supervision requires the two largest budgets to start within Fine")
    with torch.no_grad():
        coarse = outputs[low].rgb if low in outputs else model.forward_frame(frame, state, low, encoded).rgb
        indices = model.positions.order(2)[low - offset:high - offset]
        target, valid = detail_targets(frame, coarse, model.positions.shapes[2], indices)
    return (weight / len(outputs)) * suffix_angular_loss(encoded.quantized[:, low:high], target, valid)
