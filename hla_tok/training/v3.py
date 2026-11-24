"""Reconstruction-first objectives; a single source and a single encoded prefix per update."""

import random
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hla_tok import BUDGETS, Geometry, HLATok


@lru_cache(maxsize=32)
def _shuffled_block(seed: int, block: int, proportions: tuple[int, ...]) -> tuple[int, ...]:
    if len(proportions) != len(BUDGETS) or sum(proportions) != 100 or min(proportions) < 1:
        raise ValueError("each 100-update budget block must use all eight configured proportions")
    budgets = [budget for budget, count in zip(BUDGETS, proportions, strict=True) for _ in range(count)]
    random.Random(f"hla-tok-v3:{seed}:{block}").shuffle(budgets)
    return tuple(budgets)


def stratified_budget(seed: int, update: int, proportions: tuple[int, ...]) -> int:
    block, offset = divmod(update, 100)
    return _shuffled_block(seed, block, proportions)[offset]


def prefix_weights(geometry: Geometry, sampled: int, fixed: int, beta: float = 0.0) -> dict[int, float]:
    weights = {geometry.scale_budget(sampled): 1.0}
    if beta:
        budget = geometry.scale_budget(fixed)
        weights[budget] = weights.get(budget, 0.0) + beta
    return weights


def reconstruct_tokens(model: HLATok, source: Tensor, budgets: dict[int, float], *, quantize: bool):
    tokens = model.encode_training(source, max(budgets), quantize=quantize)
    features = {budget: model._decode_features(tokens.values[:, :budget], tokens.geometry) for budget in budgets}
    pixels = {budget: model.rgb_decoder(grid, tokens.geometry) for budget, grid in features.items()}
    return pixels, tokens, features


def objective(model, source, loss, crops, budgets, *, quantize=False, teacher=None, latent_weight=0.0):
    if not budgets:
        predicted = model(source)
        return loss(source, predicted, crops), predicted
    pixels, tokens, features = reconstruct_tokens(model, source, budgets, quantize=quantize)
    predicted = pixels[next(iter(budgets))]
    raw: dict[str, Tensor] = {}
    for budget, weight in budgets.items():
        terms = loss(source, pixels[budget], crops)
        for key, value in terms.items():
            raw[key] = raw[key] + value * weight if key in raw else value * weight
    if quantize:
        raw.update(vq_codebook=tokens.codebook_loss, vq_commitment=tokens.commitment_loss)
    if latent_weight:
        with torch.no_grad():
            _, reference = teacher(source, tokens.geometry)
        raw["latent"] = F.l1_loss(features[next(iter(budgets))].float(), reference.float())
    return raw, predicted


def weighted_terms(raw, weights, latent_weight):
    return {key: value * (latent_weight if key == "latent" else weights[key]) for key, value in raw.items()}


def optimize(model, source, loss, crops, budgets, optimizer, *, quantize=False, teacher=None, latent_weight=0.0,
             clip_norm=1.0, bf16=True):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=bf16):
        raw, _ = objective(model, source, loss, crops, budgets, quantize=quantize, teacher=teacher,
                           latent_weight=latent_weight)
        weighted = weighted_terms(raw, loss.weights, latent_weight)
        total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite reconstruction objective")
    total.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    metrics = {
        **{f"loss_{key}": value for key, value in raw.items()},
        **{f"weighted_{key}": value for key, value in weighted.items()},
        "gradient_l2_before_clip": norm,
    }
    return dict(zip(metrics, torch.stack(tuple(metrics.values())).detach().float().cpu().tolist(), strict=True))
