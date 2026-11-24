"""Frozen perceptual losses that preserve gradients to reconstructed RGB."""

import torch
from torch import nn
from torch.nn import functional as F

from cp_adaptok.metrics import MetricSuite
from hla_tok.progressive.geometry import NATIVE_SHAPE


class Losses(nn.Module):
    def __init__(self, device, *, rgb_l1=1.0, rgb_mse=0.1, lpips_vgg=1.0, dists=0.2, lpips_alex=0.0,
                 vq_codebook=1.0, vq_commitment=0.25):
        super().__init__()
        self.weights = dict(rgb_l1=rgb_l1, rgb_mse=rgb_mse, lpips_vgg=lpips_vgg, dists=dists,
                            vq_codebook=vq_codebook, vq_commitment=vq_commitment)
        if lpips_alex:
            self.weights["lpips_alex"] = lpips_alex
        self.alex = None
        self.lpips = None
        self.dists = None
        if lpips_vgg:
            import lpips

            self.lpips = lpips.LPIPS(net="vgg").to(device=device, memory_format=torch.channels_last)
        if lpips_alex or dists:
            names = tuple(name for name, weight in (("lpips", lpips_alex), ("dists", dists)) if weight)
            suite = MetricSuite(device, names=names)
            if suite.errors:
                raise RuntimeError(f"Missing perceptual training weights: {suite.errors}")
            if lpips_alex:
                self.alex = suite.networks["lpips"].to(memory_format=torch.channels_last)
            if dists:
                self.dists = suite.networks["dists"].to(memory_format=torch.channels_last)
        self.eval().requires_grad_(False)

    def _perceptual(self, original, predicted):
        rgb0 = original.contiguous(memory_format=torch.channels_last)
        rgb1 = predicted.contiguous(memory_format=torch.channels_last)
        with torch.autocast(rgb0.device.type, enabled=False):
            if self.alex is not None:
                yield "lpips_alex", self.alex(rgb0 * 2 - 1, rgb1 * 2 - 1).mean()
            if self.lpips is not None:
                yield "lpips_vgg", self.lpips(rgb0 * 2 - 1, rgb1 * 2 - 1).mean()
            if self.dists is not None:
                yield "dists", self.dists(rgb0, rgb1, require_grad=True).mean()

    def forward(self, original, predicted, crop_position=(0, 0)) -> dict:
        original, predicted = original.float(), predicted.float()
        terms = {"rgb_l1": F.l1_loss(predicted, original), "rgb_mse": F.mse_loss(predicted, original)}
        if self.alex is None and self.lpips is None and self.dists is None:
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


class ReconstructionLoss(Losses):
    """Full valid RGB pixel loss plus equal-weighted local and overview supervision."""

    def __init__(self, device, *, rgb_l1=1.0, rgb_mse=0.5, lpips_alex=0.5, lpips_vgg=0.5, dists=0.2,
                 vq_codebook=1.0, vq_commitment=0.25):
        super().__init__(device, rgb_l1=rgb_l1, rgb_mse=rgb_mse, lpips_alex=lpips_alex, lpips_vgg=lpips_vgg,
                         dists=dists, vq_codebook=vq_codebook, vq_commitment=vq_commitment)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, original, predicted, crop_positions=()) -> dict:
        height, width = original.shape[-2:]
        native = (height, width) == NATIVE_SHAPE
        if original.shape != predicted.shape or (not native and (height, width) != (512, 512)):
            raise ValueError("expected matched 512x512 crops or native 1080p RGB")
        if len(crop_positions) != (2 if native else 1) or any(
            not (0 <= top <= height - 256 and 0 <= left <= width - 256) for top, left in crop_positions
        ):
            raise ValueError("perceptual supervision requires the planned 256x256 crops")
        original, predicted = original.float(), predicted.float()
        with torch.autocast(original.device.type, enabled=False):
            terms = {"rgb_l1": F.l1_loss(predicted, original), "rgb_mse": F.mse_loss(predicted, original)}
            if self.alex is None and self.lpips is None and self.dists is None:
                return terms
            overview = (original, predicted)
            if native:
                overview = tuple(F.interpolate(rgb, size=(144, 256), mode="area") for rgb in overview)
            whole = dict(self._perceptual(*overview))
            local = [
                dict(self._perceptual(original[..., top:top + 256, left:left + 256],
                                      predicted[..., top:top + 256, left:left + 256]))
                for top, left in crop_positions
            ]
            return terms | {key: (value + sum(row[key] for row in local) / len(local)) * 0.5
                            for key, value in whole.items()}
