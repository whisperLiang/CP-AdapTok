import math
from math import isfinite
from pathlib import Path
from statistics import mean

import torch
from torch import Tensor
from torch.nn import functional as F

from common.integrity import sha256_file
from common.video import NATIVE_SHAPE


class MetricSuite:
    """Missing optional metric weights are explicit null results with recorded reasons."""

    def __init__(self, device, names=("psnr", "ms-ssim", "lpips", "dists")):
        self.names, self.errors, self.networks = names, {}, {}
        for name in names:
            try:
                self._load(name, device)
            except (ImportError, OSError, RuntimeError) as error:
                self.errors[name] = str(error)

    def _load(self, name, device):
        if name == "lpips":
            import lpips

            self.networks[name] = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
        elif name == "dists":
            from pathlib import Path

            import DISTS_pytorch
            from DISTS_pytorch import DISTS

            # Upstream's default looks under sys.prefix; use its distributed weight file explicitly.
            network = DISTS(load_weights=False)
            weights_path = Path(DISTS_pytorch.__file__).with_name("weights.pt")
            weights = torch.load(weights_path, map_location="cpu", weights_only=True)
            with torch.no_grad():
                network.alpha.copy_(weights["alpha"])
                network.beta.copy_(weights["beta"])
            self.networks[name] = network.to(device).eval().requires_grad_(False)
        elif name == "ms-ssim":
            from pytorch_msssim import ms_ssim

            self.networks[name] = ms_ssim
        elif name != "psnr":
            raise ValueError(f"unknown metric {name}")

    @torch.no_grad()
    def __call__(self, original, decoded):
        mse = F.mse_loss(original.float(), decoded.float()).item()
        result = {"mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12))}
        for name in self.names:
            if name == "psnr":
                continue
            if name in self.errors:
                result[name] = None
            elif name == "lpips":
                result[name] = self.networks[name](original * 2 - 1, decoded * 2 - 1).mean().item()
            elif name == "dists":
                result[name] = self.networks[name](original, decoded).mean().item()
            else:
                result[name] = self.networks[name](original, decoded, data_range=1, size_average=True).item()
        return result


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


class ReferenceDists:
    """Use the upstream distance reduction and reuse its unchanged reference features."""

    def __init__(self, network):
        self.network = network
        self.alpha, self.beta, self.chns = network.alpha, network.beta, network.chns

    def prepare(self, reference: Tensor) -> None:
        self.reference = reference
        self.reference_features = self.network.forward_once(reference)

    def forward_once(self, pixels: Tensor):
        return self.reference_features if pixels is self.reference else self.network.forward_once(pixels)

    def __call__(self, reference: Tensor, pixels: Tensor) -> Tensor:
        return type(self.network).forward(self, reference, pixels)

class NativeQuality:
    def __init__(self):
        self.suite = MetricSuite("cuda", ("psnr", "lpips", "dists"))
        if self.suite.errors:
            raise RuntimeError(f"native metrics unavailable: {self.suite.errors}")
        for name in ("lpips", "dists"):
            self.suite.networks[name].to(memory_format=torch.channels_last)
        self.dists = ReferenceDists(self.suite.networks["dists"])
        self.suite.networks["dists"] = self.dists

    def prepare(self, original: Tensor) -> None:
        if original.shape != (1, 3, *NATIVE_SHAPE):
            raise ValueError("expected one native 1080p RGB frame")
        self.original = original.float().contiguous(memory_format=torch.channels_last)
        self.dists.prepare(self.original)

    def score(self, image: Tensor) -> dict:
        if image.shape != self.original.shape:
            raise ValueError("reconstruction and source must have identical native shape")
        image = image.float().contiguous(memory_format=torch.channels_last)
        row = self.suite(self.original, image)
        if not all(isfinite(value) for value in row.values()):
            raise ValueError("nonfinite native frame quality")
        return row

def summarize(rows: list[dict]) -> dict:
    # Every frame has the same count of valid RGB values at native 1080p.
    result = {name: mean(row[name] for row in rows) for name in ("mse", "lpips", "dists")}
    return result | {"frames": len(rows), "global_psnr": psnr(result["mse"]),
                     "mean_frame_psnr": mean(row["psnr"] for row in rows)}
