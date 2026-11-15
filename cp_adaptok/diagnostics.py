"""Bounded, detached training diagnostics; never part of the transmitted representation."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def distribution_stats(features, sample_limit=512):
    flat = features.detach().float().flatten(0, -2)
    # Evenly spaced samples include the tail; sampling consumes no training RNG.
    indices = torch.linspace(0, len(flat) - 1, min(len(flat), sample_limit), device=flat.device).long()
    sample = flat[indices]
    centered = sample - sample.mean(0)
    singular = torch.linalg.svdvals(centered)
    mass = singular.square()
    probability = mass / mass.sum().clamp_min(1e-20)
    rank = (-(probability * probability.clamp_min(1e-20).log()).sum()).exp()
    normalized = F.normalize(sample[:128], dim=-1)
    similarities = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(normalized), device=flat.device, dtype=torch.bool)
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "norm_mean": flat.norm(dim=-1).mean().item(),
        "token_variance": flat.var(0, unbiased=False).mean().item(),
        "effective_rank": rank.item() if flat.var(0, unbiased=False).mean() > 1e-12 else 0.0,
        "rank_sample_count": len(sample),
        "query_cosine_mean": similarities[off_diagonal].mean().item() if len(sample) > 1 else 1.0,
    }


@torch.no_grad()
def assignment_stats(ids, size):
    counts = torch.bincount(ids.flatten(), minlength=size).float()
    probability = counts / counts.sum()
    return {
        "used_codes": int((counts > 0).sum()),
        "usage": (counts > 0).float().mean().item(),
        "perplexity": (-(probability * probability.clamp_min(1e-12).log()).sum()).exp().item(),
        "counts": counts.int().cpu().tolist(),
    }


def gradient_norms(model):
    modules = {
        "query_encoder": model.tokenizer,
        "project_in": model.vq.project_in,
        "codebook": model.vq.codebook,
        "project_out": model.vq.project_out,
        "decoder": model.decoder,
    }
    return {
        name: sum(p.grad.detach().float().square().sum() for p in part.parameters() if p.grad is not None)
        .sqrt()
        .item()
        if any(p.grad is not None for p in part.parameters())
        else 0.0
        for name, part in modules.items()
    }


@torch.no_grad()
def vq_diagnostics(vq, features, result):
    z, weight = vq.project_features(features), vq.codebook_vectors()
    flat = z.flatten(0, 1)
    nearest, distances, gaps = [], [], []
    for chunk in flat.split(vq.chunk_size):
        values, indices = vq.distances(chunk, weight).topk(2, largest=False)
        nearest.append(indices[:, 0])
        distances.append(values[:, 0])
        gaps.append((values[:, 1] - values[:, 0]) / 2)
    return {
        "query": distribution_stats(features),
        "pre_vq": distribution_stats(z),
        "nearest_distance_mean": torch.cat(distances).mean().item(),
        "top1_top2_similarity_gap": torch.cat(gaps).mean().item(),
        "deterministic": assignment_stats(torch.cat(nearest), len(weight)),
        "training": assignment_stats(result.ids, len(weight)) if result.ids is not None else None,
        "commitment_loss": result.commitment.item(),
        "codebook_loss": result.codebook_loss.item(),
    }
