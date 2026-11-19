"""Full-resolution distortion/color and high-coverage pretrained LPIPS loss."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def bt709_chroma(rgb):
    red, green, blue = rgb.chunk(3, dim=1)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return torch.cat(((blue - luma) / (2 * (1 - 0.0722)), (red - luma) / (2 * (1 - 0.2126))), 1)


def chroma_loss(original, reconstructed):
    original, reconstructed = bt709_chroma(original), bt709_chroma(reconstructed)
    loss = F.l1_loss(original, reconstructed)
    for size in (8, 32):
        loss = loss + F.l1_loss(F.avg_pool2d(original, size), F.avg_pool2d(reconstructed, size))
    return loss


def crop_plan(height, width, generator):
    result = []
    for y0 in (0, height // 2):
        for x0 in (0, width // 2):
            y = y0 + int(torch.randint(height // 2 - 256 + 1, (), generator=generator))
            x = x0 + int(torch.randint(width // 2 - 256 + 1, (), generator=generator))
            result.append((y, x))
    return result


class ReconstructionLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        import lpips

        self.perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False)
        self.weights = weights

    def perceptual_loss(self, original, reconstructed, crops):
        def distance(x, y):
            return self.perceptual(x * 2 - 1, y * 2 - 1).mean()

        global_x = F.interpolate(original, size=(270, 480), mode="bilinear", align_corners=False)
        global_y = F.interpolate(reconstructed, size=(270, 480), mode="bilinear", align_corners=False)
        local_x = torch.cat([original[..., y : y + 256, x : x + 256] for y, x in crops])
        local_y = torch.cat([reconstructed[..., y : y + 256, x : x + 256] for y, x in crops])
        return (
            checkpoint(distance, global_x, global_y, use_reentrant=False)
            + checkpoint(distance, local_x, local_y, use_reentrant=False)
        ) / 2

    def forward(self, original, reconstructed, crops):
        components = {
            "l1": F.l1_loss(original, reconstructed),
            "mse": F.mse_loss(original, reconstructed),
            "chroma": chroma_loss(original, reconstructed),
            "lpips": self.perceptual_loss(original, reconstructed, crops),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components


@torch.no_grad()
def chroma_retention(original, reconstructed):
    source = bt709_chroma(original).var((0, 2, 3), correction=0)
    target = bt709_chroma(reconstructed).var((0, 2, 3), correction=0)
    return (target / source.clamp_min(1e-10)).mean().item()
