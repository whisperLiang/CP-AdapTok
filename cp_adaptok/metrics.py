import math

import torch
from torch.nn import functional as F


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


def summarize_latency(values_ms):
    import numpy as np

    array = np.asarray(values_ms, dtype=float)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps": float(1000 / array.mean()),
    }
