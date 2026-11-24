"""Frozen perceptual losses that preserve gradients to reconstructed RGB."""

import torch
from torch import nn
from torch.nn import functional as F

from cp_adaptok.metrics import MetricSuite


class Losses(nn.Module):
    def __init__(self, device, *, rgb_l1=1.0, rgb_mse=0.1, lpips_vgg=1.0, dists=0.2,
                 vq_codebook=1.0, vq_commitment=0.25):
        super().__init__()
        self.weights = dict(rgb_l1=rgb_l1, rgb_mse=rgb_mse, lpips_vgg=lpips_vgg, dists=dists,
                            vq_codebook=vq_codebook, vq_commitment=vq_commitment)
        self.lpips = None
        self.dists = None
        if lpips_vgg:
            import lpips

            self.lpips = lpips.LPIPS(net="vgg").to(device=device, memory_format=torch.channels_last)
        if dists:
            suite = MetricSuite(device, names=("dists",))
            if suite.errors:
                raise RuntimeError(f"DISTS is required for this training configuration: {suite.errors}")
            self.dists = suite.networks["dists"].to(memory_format=torch.channels_last)
        self.eval().requires_grad_(False)

    def _perceptual(self, original, predicted):
        rgb0 = original.contiguous(memory_format=torch.channels_last)
        rgb1 = predicted.contiguous(memory_format=torch.channels_last)
        with torch.autocast(rgb0.device.type, enabled=False):
            if self.lpips is not None:
                yield "lpips_vgg", self.lpips(rgb0 * 2 - 1, rgb1 * 2 - 1).mean()
            if self.dists is not None:
                yield "dists", self.dists(rgb0, rgb1, require_grad=True).mean()

    def forward(self, original, predicted, crop_position=(0, 0)) -> dict:
        original, predicted = original.float(), predicted.float()
        terms = {"rgb_l1": F.l1_loss(predicted, original), "rgb_mse": F.mse_loss(predicted, original)}
        if self.lpips is None and self.dists is None:
            return terms
        if original.shape[-2] <= 256:
            return terms | dict(self._perceptual(original, predicted))
        top, left = crop_position
        crop = (..., slice(top, top + 256), slice(left, left + 256))
        local = dict(self._perceptual(original[crop], predicted[crop]))
        shape = (144, 256) if original.shape[-2:] == (1080, 1920) else (256, 256)
        overview = dict(self._perceptual(F.interpolate(original, size=shape, mode="area"),
                                         F.interpolate(predicted, size=shape, mode="area")))
        return terms | {key: (local[key] + overview[key]) * 0.5 for key in local}
