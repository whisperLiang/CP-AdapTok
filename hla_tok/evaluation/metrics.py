"""Matched native RGB quality using the CP perceptual weights and seam definition."""

import hashlib
import math
from pathlib import Path
from statistics import mean

import torch

from cp_adaptok.metrics import MetricSuite
from hla_tok.progressive.geometry import NATIVE_SHAPE, REGION_SIZE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_hashes() -> dict[str, str]:
    import DISTS_pytorch
    import lpips

    hub = Path(torch.hub.get_dir()) / "checkpoints"
    heads = Path(lpips.__file__).parent / "weights/v0.1"
    paths = {
        "imagenet_alexnet": hub / "alexnet-owt-7be5be79.pth",
        "imagenet_vgg16": hub / "vgg16-397923af.pth",
        "lpips_alex_v01": heads / "alex.pth",
        "lpips_vgg_v01": heads / "vgg.pth",
        "dists": Path(DISTS_pytorch.__file__).with_name("weights.pt"),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def psnr(mse: float) -> float:
    return -10.0 * math.log10(mse) if mse else float("inf")


def boundary_mask(height: int, width: int, device, half_width: int = 2):
    # Original AdapTok counts [boundary-2, boundary+2), excluding image edges.
    def axis(length):
        coord = torch.arange(length, device=device)
        previous = (coord // REGION_SIZE) * REGION_SIZE
        following = previous + REGION_SIZE
        return ((previous > 0) & (coord - previous < half_width)) | (
            (following < length) & (following - coord <= half_width)
        )

    return axis(height)[:, None] | axis(width)[None, :]


class FullFrameQuality:
    def __init__(self, device, hashes: dict[str, str]):
        import lpips

        self.weight_hashes = metric_hashes()
        if self.weight_hashes != hashes:
            raise ValueError("quality weights differ from frozen provenance")
        self.suite = MetricSuite(device)
        if self.suite.errors:
            raise RuntimeError(f"missing required full-frame metrics: {self.suite.errors}")
        for name in ("lpips", "dists"):
            self.suite.networks[name].to(memory_format=torch.channels_last)
        self.vgg = lpips.LPIPS(net="vgg").to(device=device, memory_format=torch.channels_last)
        self.vgg.eval().requires_grad_(False)
        self.mask = boundary_mask(*NATIVE_SHAPE, device)

    @torch.inference_mode()
    def __call__(self, original, reconstructed):
        if original.shape != reconstructed.shape or original.shape[-2:] != NATIVE_SHAPE:
            raise ValueError("quality metrics require matched native 1080p frames")
        original = original.float().contiguous(memory_format=torch.channels_last)
        reconstructed = reconstructed.float().contiguous(memory_format=torch.channels_last)
        metrics = self.suite(original, reconstructed)
        metrics["lpips_vgg"] = self.vgg(original * 2 - 1, reconstructed * 2 - 1).mean().item()
        error = (original - reconstructed).square()
        metrics["boundary_mse"] = error[..., self.mask].mean().item()
        metrics["interior_mse"] = error[..., ~self.mask].mean().item()
        metrics["boundary_minus_interior_mse"] = metrics["boundary_mse"] - metrics["interior_mse"]
        return metrics


def summarize_quality(rows: list[dict]) -> dict:
    fields = ("mse", "psnr", "lpips", "lpips_vgg", "dists", "ms-ssim",
              "boundary_mse", "interior_mse", "boundary_minus_interior_mse")
    values = {field: mean(row[field] for row in rows) for field in fields}
    frame_psnr = values.pop("psnr")
    return values | {
        "frames": len(rows),
        "global_psnr": psnr(values["mse"]),
        "mean_frame_psnr": frame_psnr,
        "boundary_psnr": psnr(values["boundary_mse"]),
        "interior_psnr": psnr(values["interior_mse"]),
    }
