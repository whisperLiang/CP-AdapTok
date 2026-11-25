"""The V3 RGB metrics, with the reference DISTS features prepared once per frame."""

from math import isfinite
from statistics import mean

import torch
from torch import Tensor

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.metrics import psnr
from hla_tok.progressive.geometry import NATIVE_SHAPE


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
            raise RuntimeError(f"V3 metrics unavailable: {self.suite.errors}")
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
