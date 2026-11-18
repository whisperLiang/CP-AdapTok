"""Supervision sees originals; model references always come from real VQ reconstructions."""

import torch
from torch import nn
from torch.nn import functional as F


def high_frequency_l1(rgb, target):
    # The high-pass is linear. Filter the difference once in FP32. Slice-based
    # reflection also supports a deterministic backward on native CUDA frames.
    delta = rgb.float() - target.float()
    wide = torch.cat((delta[..., 1:2], delta, delta[..., -2:-1]), dim=3)
    padded = torch.cat((wide[:, :, 1:2], wide, wide[:, :, -2:-1]), dim=2)
    return (delta - F.avg_pool2d(padded, 3, 1)).abs().mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, weights, device, settings=None):
        super().__init__()
        self.weights = weights
        self.settings = settings or {}
        self.mode = self.settings.get("mode", "legacy")
        self.network = self.settings.get("network", "alex")
        self.validate_objective()
        self.crop_rng = (torch.Generator().manual_seed(self.settings["crop_seed"])
                         if "crop_seed" in self.settings else None)
        self.perceptual = None
        if weights.get("perceptual", 0):
            import lpips

            self.perceptual = lpips.LPIPS(net=self.network).to(device).eval().requires_grad_(False)

    def validate_objective(self):
        if self.mode not in ("legacy", "unified") or self.network not in ("alex", "vgg"):
            raise ValueError("unknown reconstruction mode or LPIPS network")
        if self.mode == "unified":
            if not self.unified_weights_valid():
                raise ValueError("unified requires RGB L1 + LPIPS + VQ only")
            if self.weights.get("perceptual", 0) and "crop_seed" not in self.settings:
                raise ValueError("unified LPIPS requires an independent crop_seed")

    def unified_weights_valid(self):
        unsupported = [value for key, value in self.weights.items() if key not in ("l1", "perceptual", "vq")]
        return self.weights.get("l1") == 1 and not any(unsupported)

    def sample_views(self, target):
        if self.crop_rng is None:
            return None
        h, w = target.shape[-2:]
        size = self.settings.get("crop_size", 256)
        if not 1 <= size <= min(h, w):
            raise ValueError("native LPIPS crop exceeds frame")
        y = int(torch.randint(h - size + 1, (), generator=self.crop_rng))
        x = int(torch.randint(w - size + 1, (), generator=self.crop_rng))
        return [(x, y, size, size)]

    def perception(self, rgb, target, views):
        if self.perceptual is None:
            return rgb.new_zeros(()), {}
        with torch.autocast(rgb.device.type, enabled=False):
            if self.mode == "legacy":
                return self.perceptual(rgb * 2 - 1, target * 2 - 1).mean(), {}
            if views is None:
                raise ValueError("unified LPIPS needs shared frame crop coordinates")
            size = self.settings.get("global_size", [270, 480])
            small = [F.interpolate(v, size=size, mode="bilinear", align_corners=False) for v in (rgb, target)]
            global_loss = self.perceptual(small[0] * 2 - 1, small[1] * 2 - 1).mean()
            crops = [self.perceptual(rgb[..., y:y+h, x:x+w] * 2 - 1,
                                     target[..., y:y+h, x:x+w] * 2 - 1).mean() for x, y, w, h in views]
            native_loss = torch.stack(crops).mean()
        return (global_loss + native_loss) * .5, {"perceptual_global": global_loss,
                                                "perceptual_native_crop": native_loss}

    def distortion(self, rgb, target, views=None):
        rgb, target = rgb.float(), target.float()
        mse, l1 = F.mse_loss(rgb, target), F.l1_loss(rgb, target)
        perception, view_terms = self.perception(rgb, target, views)
        loss = self.weights.get("mse", 0) * mse + self.weights.get("l1", 0) * l1
        loss = loss + self.weights.get("perceptual", 0) * perception
        terms = {"mse": mse, "l1": l1, "perceptual": perception, **view_terms}
        if self.weights.get("detail", 0):
            terms["detail"] = high_frequency_l1(rgb, target)
            loss = loss + self.weights["detail"] * terms["detail"]
        return loss, terms

    def temporal(self, rgb, target, state, previous_target):
        if state is None or previous_target is None:
            return rgb.new_zeros(())
        return self.weights.get("temporal", 0) * F.l1_loss(
            rgb.float() - state.rgb.float(), target.float() - previous_target.float())
