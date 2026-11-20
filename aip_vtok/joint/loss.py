"""Native RGB distortion, differentiable LPIPS/DISTS views, latent and reference losses."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aip_vtok.training.losses import chroma_loss, native_global_view

from .bitstream import PIXELS


class JointLoss(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        from DISTS_pytorch import DISTS
        from lpips import LPIPS

        self.lpips = LPIPS(net="alex").eval().requires_grad_(False)
        self.dists = DISTS().eval().requires_grad_(False)
        self.weights = weights

    def perceptual(self, original: torch.Tensor, reconstructed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lpips = self.lpips(original * 2 - 1, reconstructed * 2 - 1).mean()
        dists = self.dists(original, reconstructed, require_grad=True, batch_average=True)
        return lpips, dists

    def views(self, original: torch.Tensor, reconstructed: torch.Tensor, crops: list[tuple[int, int]]):
        def split(x):
            return torch.cat([x[..., y:y + 256, h:h + 256] for y, h in crops])

        global_metrics = checkpoint(self.perceptual, native_global_view(original), native_global_view(reconstructed),
                                    use_reentrant=False)
        local_metrics = checkpoint(self.perceptual, split(original), split(reconstructed), use_reentrant=False)
        return tuple((a + b) * 0.5 for a, b in zip(global_metrics, local_metrics, strict=True))

    def forward(self, rgb, rec, target, latent, bits, crops):
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rec, rgb = rec.float(), rgb.float()
            lpips, dists = self.views(rgb, rec, crops)
            terms = {"l1": F.l1_loss(rgb, rec), "mse": F.mse_loss(rgb, rec),
                     "chroma": chroma_loss(rgb, rec), "lpips": lpips, "dists": dists,
                     "latent": F.l1_loss(latent.float(), target.float()), "rate": bits.float().sum() / PIXELS}
        return terms
