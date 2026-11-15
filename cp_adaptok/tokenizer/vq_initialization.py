"""Reproducible real-sample clustering; insufficient data is an error."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def kmeans_centers(samples, count, seed, iterations=20, normalized=True):
    samples = samples.detach().float().cpu().flatten(0, -2)
    if not torch.isfinite(samples).all() or iterations < 1:
        raise ValueError("invalid clustering samples/iterations")
    samples = F.normalize(samples, dim=-1) if normalized else samples
    samples = samples.unique(dim=0)
    if len(samples) < count:
        raise ValueError(f"need {count} distinct real samples, received {len(samples)}")
    generator = torch.Generator().manual_seed(seed)
    centers = samples[torch.randperm(len(samples), generator=generator)[:count]].clone()
    for _ in range(iterations):
        sums, counts = torch.zeros_like(centers), torch.zeros(count)
        for chunk in samples.split(2048):
            distances = (
                chunk.square().sum(-1, keepdim=True) + centers.square().sum(-1) - 2 * chunk @ centers.T
            )
            ids = distances.argmin(-1)
            sums.index_add_(0, ids, chunk)
            counts += torch.bincount(ids, minlength=count)
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]
        centers = F.normalize(centers, dim=-1) if normalized else centers
    if len(centers.unique(dim=0)) != count:
        raise ValueError("clustering did not produce distinct centers; collect more varied real samples")
    return centers
