"""Fast matched metric evaluation that caches only identical source-image features."""

from collections import defaultdict

import lpips
import torch
from torch.nn import functional as F

from aip_vtok.evaluation.fix_metrics import fix_chroma
from aip_vtok.evaluation.gates import psnr
from cp_adaptok.metrics import MetricSuite

QUALITY = ("psnr", "lpips", "dists")


def metric_suite():
    suite = MetricSuite("cuda:0", names=("psnr", "lpips", "dists"))
    if suite.errors:
        raise RuntimeError(f"required full-resolution metrics unavailable: {suite.errors}")
    for net in suite.networks.values():
        net.to(memory_format=torch.channels_last)
    return suite


def alex_features(net, rgb):
    scaled = net.scaling_layer(rgb.to(memory_format=torch.channels_last) * 2 - 1)
    return [lpips.normalize_tensor(a) for a in net.net(scaled)]


def image_moments(x):
    mean = x.mean((2, 3), keepdim=True)
    return mean, (x - mean).square().mean((2, 3), keepdim=True)


class FrameMetrics:
    """Same full-frame AlexNet and VGG scores as MetricSuite on a shared RGB target."""

    @torch.no_grad()
    def __init__(self, suite, rgb):
        self.rgb = rgb.to(memory_format=torch.channels_last)
        self.alex, self.vgg = (suite.networks[name] for name in ("lpips", "dists"))
        self.alex_target = alex_features(self.alex, self.rgb)
        self.vgg_target = self.vgg.forward_once(self.rgb)
        self.vgg_moments = [image_moments(x) for x in self.vgg_target]
        total = self.vgg.alpha.sum() + self.vgg.beta.sum()
        self.alpha = (self.vgg.alpha / total).split(self.vgg.chns, dim=1)
        self.beta = (self.vgg.beta / total).split(self.vgg.chns, dim=1)

    def lpips(self, rec):
        features = alex_features(self.alex, rec)
        return sum(head((x - y).square()).mean((2, 3), keepdim=True)
                   for head, x, y in zip(self.alex.lins, self.alex_target, features)).mean()

    def dists(self, rec):
        eps = 1e-6
        location, structure = 0, 0
        features = self.vgg.forward_once(rec)
        for x, (mx, vx), y, alpha, beta in zip(
            self.vgg_target, self.vgg_moments, features, self.alpha, self.beta,
        ):
            my, vy = image_moments(y)
            cov = (x * y).mean((2, 3), keepdim=True) - mx * my
            location = location + (alpha * (2 * mx * my + eps) / (mx.square() + my.square() + eps)).sum(1)
            structure = structure + (beta * (2 * cov + eps) / (vx + vy + eps)).sum(1)
        return (1 - location - structure).mean()

    @torch.no_grad()
    def __call__(self, rec, latent, target):
        rec = rec.to(memory_format=torch.channels_last)
        mse = F.mse_loss(rec, self.rgb).item()
        return {"mse": mse, "psnr": psnr(mse), "lpips": self.lpips(rec).item(), "dists": self.dists(rec).item(),
                "latent_mse": F.mse_loss(latent, target).item(), **fix_chroma(self.rgb, rec)}


def regressions(values):
    worse = [v for v in values if v > 1e-6]
    return {"pairs": len(values), "violations": len(worse), "fraction": len(worse) / len(values),
            "mean_backslide": sum(worse) / max(1, len(worse)), "max_backslide": max(worse, default=0.0)}


def quality_backslides(rows, keys):
    frames = defaultdict(list)
    for row in rows:
        frames[tuple(row[key] for key in (*keys, "video", "frame"))].append(row)
    deltas = defaultdict(list)
    for label, frame_rows in frames.items():
        ordered = sorted(frame_rows, key=lambda row: row["budget"])
        for lo, hi in zip(ordered, ordered[1:]):
            for metric in QUALITY:
                change = hi[metric] - lo[metric]
                deltas[(*label[:-2], lo["budget"], hi["budget"], metric)].append(
                    -change if metric == "psnr" else change
                )
    return [{**dict(zip((*keys, "lower", "upper", "metric"), label)), **regressions(values)}
            for label, values in deltas.items()]
