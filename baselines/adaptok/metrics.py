"""Full-frame RGB metrics; diagnostic boundary strips do not replace full-frame metrics."""

import math

import torch

from common.metrics import MetricSuite

from .video_reader import file_sha256


def psnr(mse):
    return -10 * math.log10(mse) if mse > 0 else float("inf")


def boundary_mask(height, width, spacing=128, half_width=2, device="cpu"):
    if spacing < 1 or half_width < 1:
        raise ValueError("positive boundary spacing and half width required")
    mask = torch.zeros(height, width, device=device, dtype=torch.bool)
    for y in range(spacing, height, spacing):
        mask[max(0, y - half_width):min(height, y + half_width), :] = True
    for x in range(spacing, width, spacing):
        mask[:, max(0, x - half_width):min(width, x + half_width)] = True
    return mask


def pixel_metrics(original, reconstructed, mask):
    error = (original.float() - reconstructed.float()).square()
    mse = error.mean().item()
    boundary = error[..., mask].mean().item() if bool(mask.any()) else None
    interior = error[..., ~mask].mean().item() if bool((~mask).any()) else None
    return {"mse": mse, "psnr": psnr(mse), "boundary_mse": boundary, "interior_mse": interior,
            "boundary_psnr": psnr(boundary) if boundary is not None else None,
            "interior_psnr": psnr(interior) if interior is not None else None,
            "boundary_minus_interior_mse": boundary - interior if boundary is not None and interior is not None else None}


class FullFrameMetrics:
    def __init__(self, device):
        self.suite = MetricSuite(device)
        if self.suite.errors:
            raise RuntimeError(f"required CP metric unavailable: {self.suite.errors}")
        import lpips

        self.vgg = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, original, reconstructed, mask):
        if original.shape != reconstructed.shape or tuple(original.shape[-2:]) != (1080, 1920):
            raise ValueError("quality metrics require complete native 1080p RGB frames")
        result = self.suite(original, reconstructed)
        result.update(pixel_metrics(original, reconstructed, mask))
        result["lpips-vgg"] = self.vgg(original * 2 - 1, reconstructed * 2 - 1).mean().item()
        return result

    def manifest(self):
        import importlib.metadata
        from pathlib import Path

        import DISTS_pytorch
        import lpips

        paths = [Path(DISTS_pytorch.__file__).with_name("weights.pt"),
                 Path(lpips.__file__).parent / "weights/v0.1/alex.pth",
                 Path(lpips.__file__).parent / "weights/v0.1/vgg.pth"]
        paths += [Path(torch.hub.get_dir()) / "checkpoints" / p for p in ["alexnet-owt-7be5be79.pth", "vgg16-397923af.pth"]]
        return {"versions": {p: importlib.metadata.version(p) for p in ["lpips", "DISTS-pytorch", "pytorch-msssim"]},
                "weight_sha256": {str(p): file_sha256(p) for p in paths},
                "lpips_preprocess": "RGB [0,1] -> [-1,1]; LPIPS v0.1 alex and vgg",
                "dists_preprocess": "RGB [0,1], upstream DISTS internal normalization, CP distributed alpha/beta",
                "ms_ssim": "pytorch_msssim.ms_ssim(data_range=1, size_average=True)",
                "resolution": [1080, 1920], "dtype": "float32", "metric_batch_size": 1}


def aggregate(rows):
    if not rows:
        raise ValueError("cannot aggregate empty measurements")
    keys = ["mse", "ms-ssim", "lpips", "lpips-vgg", "dists", "boundary_mse", "interior_mse", "boundary_minus_interior_mse"]
    result = {k: sum(r[k] for r in rows) / len(rows) for k in keys if all(r.get(k) is not None for r in rows)}
    result.update(frames=len(rows), psnr_mean_db=sum(r["psnr"] for r in rows) / len(rows),
                  psnr_from_global_mse_db=psnr(result["mse"]), bpp=None, quality_only=True)
    for region in ["boundary", "interior"]:
        if f"{region}_mse" in result:
            result[f"{region}_psnr_from_mse_db"] = psnr(result[f"{region}_mse"])
    return result
