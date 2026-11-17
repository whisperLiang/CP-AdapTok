"""Supervision sees originals; model references always come from real VQ reconstructions."""

import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self.weights = weights
        self.perceptual = None
        if weights.get("perceptual", 0):
            import lpips

            self.perceptual = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)

    def distortion(self, rgb, target):
        rgb, target = rgb.float(), target.float()
        mse, l1 = F.mse_loss(rgb, target), F.l1_loss(rgb, target)
        perception = rgb.new_zeros(())
        if self.perceptual is not None:
            with torch.autocast(rgb.device.type, enabled=False):
                perception = self.perceptual(rgb * 2 - 1, target * 2 - 1).mean()
        loss = self.weights.get("mse", 0) * mse + self.weights.get("l1", 0) * l1
        loss = loss + self.weights.get("perceptual", 0) * perception
        return loss, {"mse": mse, "l1": l1, "perceptual": perception}

    def temporal(self, rgb, target, state, previous_target):
        if state is None or previous_target is None:
            return rgb.new_zeros(())
        return self.weights.get("temporal", 0) * F.l1_loss(
            rgb.float() - state.rgb.float(), target.float() - previous_target.float())
