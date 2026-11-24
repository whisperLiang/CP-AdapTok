"""Shared forward/backward step and deterministic train-only VQ calibration."""

from collections import defaultdict

import torch
from torch import nn

from hla_tok.models.codec import HLATok
from hla_tok.models.quantizer import VectorQuantizer
from hla_tok.training.losses import Losses


@torch.no_grad()
def initialize_codebook(vq: VectorQuantizer, observations, seed: int, iterations: int = 4):
    size = vq.embedding.num_embeddings
    if observations.ndim != 2 or len(observations) < size or observations.shape[-1] != vq.embedding.embedding_dim:
        raise ValueError("codebook fitting needs at least one candidate per code and the correct dimension")
    points = observations.detach().cpu().float()
    if not torch.isfinite(points).all():
        raise ValueError("nonfinite codebook fitting observations")
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(points), generator=generator)[:size]
    centers = points[selected].clone()
    device = vq.embedding.weight.device
    vectors = points.to(device)
    for _ in range(iterations):
        vq.embedding.weight.copy_(centers.to(device))
        nearest = vq(vectors).cpu()
        counts = torch.bincount(nearest, minlength=size)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, nearest, points)
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
    vq.embedding.weight.copy_(centers.to(device))


def _norms(gradients, grouped_names):
    groups = defaultdict(list)
    for name, grad in zip(grouped_names, gradients):
        if grad is not None:
            groups[name].append(grad.detach().float().square().sum())
    return {name: torch.stack(groups[name]).sum().sqrt().item() if groups[name] else 0.0
            for name in dict.fromkeys(grouped_names)}


def _audit_terms(weighted, model):
    names, params = zip(*model.named_parameters())
    groups = [name.split(".")[0] for name in names]
    result = {}
    for key, scalar in weighted.items():
        gradients = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True) if scalar.requires_grad else ()
        result[key] = _norms(gradients, groups)
    return result


def _code_usage(ids, size: int):
    if ids is None:
        return None
    counts = torch.bincount(ids.detach().flatten(), minlength=size)
    hits = counts[counts > 0].float()
    probabilities = hits / hits.sum()
    entropy = -(probabilities * probabilities.log2()).sum().item()
    return {"active_ratio": len(hits) / size, "dead_ratio": 1 - len(hits) / size, "entropy_bits": entropy,
            "perplexity": 2**entropy, "top_code_fraction": probabilities.max().item()}


def optimize_batch(model: HLATok, source, budgets: tuple[int, ...], optimizer, losses: Losses, *,
                   quantize=True, bf16=False, audit=False, crop_position=(0, 0), clip_norm=1.0):
    optimizer.zero_grad(set_to_none=True)
    combined = defaultdict(list)
    with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=bf16):
        tokens = model.encode_training(source, max(budgets), quantize=quantize)
        for budget in budgets:
            predicted = model.decode_training(tokens, budget)
            for key, term in losses(source, predicted, crop_position).items():
                combined[key].append(term)
    raw = {key: torch.stack(terms).mean() for key, terms in combined.items()}
    raw |= {"vq_codebook": tokens.codebook_loss, "vq_commitment": tokens.commitment_loss}
    weighted = {key: value * losses.weights[key] for key, value in raw.items()}
    scale = torch.stack(list(weighted.values()))
    if not torch.isfinite(scale).all():
        bad = [name for name, value in raw.items() if not torch.isfinite(value)]
        raise FloatingPointError(f"nonfinite training loss: {bad}")
    details = _audit_terms(weighted, model) if audit else {}
    scale.sum().backward()
    params = list(model.named_parameters())
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    gradient = _norms([p.grad for _, p in params], [k.split(".")[0] for k, _ in params]) if audit else {}
    optimizer.step()
    values = torch.stack([v.detach().float() for v in (*raw.values(), *weighted.values())]).cpu().tolist()
    original, scaled = values[:len(raw)], values[len(raw):]
    return {
        "local_budgets": list(budgets), "grad_norm": gradient_norm.item(), "finite": True,
        "module_gradient_l2": gradient,
        "losses": {key: {"raw": v, "weighted": w, "gradient_l2": details.get(key)}
                   for key, v, w in zip(raw, original, scaled)},
        "codebook": _code_usage(tokens.ids, model.quantizer.embedding.num_embeddings),
    }
