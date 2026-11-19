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


def native_global_view(rgb):
    if rgb.shape[-2:] != (1080, 1920):
        raise ValueError("native global perceptual view requires 1920x1080 RGB")
    # Exact align_corners=False bilinear centers for a factor-four downsample.
    # Disjoint slices avoid CUDA's nondeterministic interpolation backward.
    dtype = rgb.dtype
    rgb = rgb.float()
    return (
        (rgb[..., 1::4, 1::4] + rgb[..., 1::4, 2::4] + rgb[..., 2::4, 1::4] + rgb[..., 2::4, 2::4])
        .mul(0.25)
        .to(dtype)
    )


class ReconstructionLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        import lpips

        self.perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False)
        self.weights = weights

    def perceptual_loss(self, original, reconstructed, crops):
        def distance(x, y):
            return self.perceptual(x * 2 - 1, y * 2 - 1).mean()

        global_x = native_global_view(original)
        global_y = native_global_view(reconstructed)
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
        }
        if crops is not None:
            components["lpips"] = self.perceptual_loss(original, reconstructed, crops)
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components


def latent_losses(raw, fused, native):
    target = native.detach().float()
    return {
        "latent_raw": F.l1_loss(raw.float(), target),
        "latent_fused": F.l1_loss(fused.float(), target),
    }


@torch.no_grad()
def latent_mse_metrics(raw, fused, native):
    raw_mse = F.mse_loss(raw.float(), native.float()).item()
    fused_mse = F.mse_loss(fused.float(), native.float()).item()
    return {
        "raw_latent_mse": raw_mse,
        "fused_latent_mse": fused_mse,
        "delta_fusion_mse": fused_mse - raw_mse,
    }


def reference_loss(backbone, encoded, reference):
    with torch.no_grad():
        _, native_feature = backbone.decode_aligned(encoded.native_latent, encoded.context)
    return F.l1_loss(reference.feature.float(), native_feature.float())


@torch.no_grad()
def chroma_retention(original, reconstructed):
    source = bt709_chroma(original).var((0, 2, 3), correction=0)
    target = bt709_chroma(reconstructed).var((0, 2, 3), correction=0)
    return (target / source.clamp_min(1e-10)).mean().item()


@torch.no_grad()
def chroma_diagnostics(original, reconstructed):
    source, target = bt709_chroma(original), bt709_chroma(reconstructed)
    source_variance = source.var((0, 2, 3), correction=0)
    target_variance = target.var((0, 2, 3), correction=0)
    return {
        "chroma_retention": (target_variance / source_variance.clamp_min(1e-10)).mean().item(),
        "chroma_mse": F.mse_loss(source, target).item(),
        "chroma_mean_abs_bias": (source.mean((0, 2, 3)) - target.mean((0, 2, 3))).abs().mean().item(),
        "source_chroma_variance": source_variance.mean().item(),
    }
