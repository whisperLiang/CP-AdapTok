"""Train-only RGB gradient calibration; calibration never changes trial weights."""

from math import sqrt
from statistics import mean

import torch
from torch import nn

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.training.step import _norms
from hla_tok.training.v3 import objective, weighted_terms
from scripts.hla_tok.v3.protocol import CROP, NATIVE, SamplePlan

PIXEL = ("rgb_l1", "rgb_mse")
PERCEPTUAL = ("lpips_alex", "lpips_vgg", "dists")


def _rgb_inner(left, right):
    return torch.dot(left.flatten(), right.flatten()).item()


def _gradient_terms(raw, predicted, named):
    params = [tensor for _, tensor in named]
    groups = [key.split(".")[0] for key, _ in named]
    vectors, norms = {}, {}
    for name, term in raw.items():
        rgb, *grads = torch.autograd.grad(term, (predicted, *params), allow_unused=True, retain_graph=True)
        if name != "latent" and (rgb is None or not torch.isfinite(rgb).all() or not rgb.count_nonzero()):
            raise FloatingPointError(f"missing or nonfinite RGB gradient: {name}")
        if any(not grad.isfinite().all() for grad in grads if grad is not None):
            raise FloatingPointError(f"nonfinite module gradient: {name}")
        if rgb is not None:
            vectors[name] = rgb.detach().float()
        norms[name] = _norms(grads, groups)
    return vectors, norms


def _audit_sample(model, source, loss, sample, budgets, teacher, latent_weight):
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw, predicted = objective(model, source, loss, sample["perceptual_crops_yx"], budgets, teacher=teacher,
                                   latent_weight=latent_weight)
        weighted = weighted_terms(raw, loss.weights, latent_weight)
    if not all(term.isfinite() for term in raw.values()):
        raise FloatingPointError("nonfinite calibration losses")
    vectors, module_norms = _gradient_terms(raw, predicted, named)
    norm = {key: sqrt(_rgb_inner(grad, grad)) for key, grad in vectors.items()}
    cosines = {f"{left}:{right}": _rgb_inner(vectors[left], vectors[right]) / (norm[left] * norm[right])
               for index, left in enumerate(vectors) for right in list(vectors)[index + 1:]}
    pixel = sum(vectors[key] * loss.weights[key] for key in PIXEL)
    perceptual = sum(vectors[key] * loss.weights[key] for key in PERCEPTUAL)
    return {
        "sample": sample, "finite": True, "raw": {key: value.item() for key, value in raw.items()},
        "weighted_initial": {key: value.item() for key, value in weighted.items()},
        "rgb_gradient_l2": norm, "rgb_gradient_cosines": cosines,
        "module_gradient_l2_unweighted": module_norms,
        "pixel_gradient_square": _rgb_inner(pixel, pixel),
        "perceptual_gradient_square": _rgb_inner(perceptual, perceptual),
    }


def _measure_update(model, source, loss, sample, config, budgets, teacher, latent_weight):
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    initial = {name: param.detach().clone() for name, param in params.items()}
    rate = config["native_learning_rate"] if source.shape[-2:] == (NATIVE.height, NATIVE.width) else config["learning_rate"]
    optimizer = torch.optim.AdamW(params.values(), lr=rate / config["warmup_updates"],
                                  weight_decay=config["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw, _ = objective(model, source, loss, sample["perceptual_crops_yx"], budgets, teacher=teacher,
                           latent_weight=latent_weight)
        terms = weighted_terms(raw, loss.weights, latent_weight)
    sum(terms.values()).backward()
    clipped = nn.utils.clip_grad_norm_(params.values(), config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    groups = [name.split(".")[0] for name in params]
    with torch.no_grad():
        changes = [param - initial[name] for name, param in params.items()]
        for name, param in params.items():
            param.copy_(initial[name])
    after = _norms(changes, groups)
    if not all(value > 0 for value in after.values()):
        raise FloatingPointError("an applicable module did not update during calibration")
    return {"gradient_l2_before_clip": clipped.item(), "actual_parameter_update_l2": after,
            "weighted_final": {key: value.item() for key, value in terms.items()}}


def calibration(model, reader, loss, config, destination, *, teacher=None, freeze_weights=False):
    controls = config["gradient_calibration"]
    count = controls["samples"]
    plan = SamplePlan(reader, [CROP] * (count // 2) + [NATIVE] * (count - count // 2),
                      controls["seed_offset"], config)
    prepared, rows = [], []
    for raw, geometry, _, sample in plan.batches(0, count):
        budgets = {} if teacher is None else {geometry.scale_budget(config["native_budget"]): 1.0}
        latent_weight = config["latent"]["weight"] if teacher is not None else 0.0
        rgb = transfer(raw[None])
        rows.append(_audit_sample(model, rgb, loss, sample, budgets, teacher, latent_weight))
        prepared.append((raw, sample, budgets, latent_weight))
    ratio = sqrt(mean(row["perceptual_gradient_square"] for row in rows)
                 / mean(row["pixel_gradient_square"] for row in rows))
    if not 0 < ratio < float("inf"):
        raise FloatingPointError("invalid initial perceptual/pixel RGB gradient ratio")
    low, high = controls["ratio_interval"]
    minimum, maximum = controls["multiplier_interval"]
    multiplier = 1.0 if freeze_weights or low <= ratio <= high else min(maximum, max(minimum, 1.0 / ratio))
    original = dict(loss.weights)
    loss.weights.update({key: original[key] * multiplier for key in PERCEPTUAL})
    for (raw, sample, budgets, latent_weight), row in zip(prepared, rows, strict=True):
        row.update(_measure_update(model, transfer(raw[None]), loss, sample, config, budgets, teacher, latent_weight))
    model.zero_grad(set_to_none=True)
    write_json(destination, {"initial_weights": original, "weights": dict(loss.weights), "samples": rows,
                             "initial_perceptual_pixel_ratio": ratio, "perceptual_multiplier": multiplier,
                             "calibrated_ratio": ratio * multiplier, "sample_chain": plan.chains[-1],
                             "freeze_weights": freeze_weights, "adjustment": "clip(1/ratio) outside registered band"})
