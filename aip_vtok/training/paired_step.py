"""One EncodeOnce and one receiver history context for all three training budgets."""

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.training.budgets import sample_budgets
from aip_vtok.training.diagnostics import loss_record
from aip_vtok.training.losses import crop_plan, latent_losses, latent_mse_metrics, reference_loss
from aip_vtok.training.prefix import (
    commit_training_reference,
    prefix_latents,
    receiver_context,
    render_training,
    training_reference,
)


def anchored_budget(codec, encoded, objective, rgb, crops, budget, weights, inspect):
    raw, fused = prefix_latents(codec, encoded.tokens, encoded.prediction, budget)
    rec, _ = codec.backbone.decode_aligned(fused, encoded.context, checkpointed=True)
    rgb_loss, rgb_terms = objective(rgb, rec, crops)
    alignment = latent_losses(raw, fused, encoded.native_latent)
    loss = rgb_loss + sum(weights[name] * value for name, value in alignment.items())
    record = {
        "frame": encoded.frame_id,
        "budget": budget,
        "components": {name: float(value.detach()) for name, value in (rgb_terms | alignment).items()},
        **latent_mse_metrics(raw, fused, encoded.native_latent),
    }
    if inspect:
        record["weighted_grad_norm_at_latent"] = {}
        for name, value in (rgb_terms | alignment).items():
            boundary = raw if name == "latent_raw" else fused
            weight = (weights | objective.weights)[name]
            norm = torch.autograd.grad(weight * value, boundary, retain_graph=True)[0].float().norm()
            record["weighted_grad_norm_at_latent"][name] = float(norm)
    return loss, record


def anchored_frame(codec, encoded, objective, rgb, crops, budgets, weights, inspect):
    measured = [
        anchored_budget(codec, encoded, objective, rgb, crops, budget, weights, inspect and i == 0)
        for i, budget in enumerate(budgets)
    ]
    losses, records = zip(*measured)
    reference = training_reference(codec, encoded)
    loss = reference_loss(codec.backbone, encoded, reference)
    weighted = weights["reference"] * loss
    records[0]["reference_loss"] = float(loss.detach())
    records[0]["reference_budget"] = reference.budget
    records[0]["tanh_saturation_ratio"] = float(encoded.tokens.saturation.float().mean())
    if inspect:
        grad = torch.autograd.grad(weighted, reference.latent, retain_graph=True)[0]
        records[0]["weighted_grad_norm_at_latent"]["reference"] = float(grad.float().norm())
    return torch.stack(losses).mean() + weighted, records, reference


def paired_step(
    codec, clip, objective, rng, inspect_gradients=False, *, weights=None, perceptual_frames=None
):
    state, total, records = ReferenceState(), 0.0, []
    before = codec.encode_calls
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        encoded = codec.encode_frame(rgb, state, frame_id)
        crops = crop_plan(1080, 1920, rng.crop_rng)
        budgets = sample_budgets(rng)
        inspect = inspect_gradients and frame_id == 0
        if weights is not None:
            if perceptual_frames is not None and frame_id < len(clip) - perceptual_frames:
                crops = None
            loss, measured, reference = anchored_frame(
                codec, encoded, objective, rgb, crops, budgets, weights, inspect
            )
            context, prediction = encoded.context, encoded.prediction
        else:
            context, prediction = receiver_context(codec, state, frame_id, encoded.tokens)
            loss, measured = legacy_frame(
                codec, encoded, objective, rgb, crops, budgets, context, prediction, inspect
            )
            reference = None
        records.extend(measured)
        loss = loss / len(clip)
        loss.backward()
        total += float(loss.detach())
        state = commit_training_reference(codec, encoded.tokens, prediction, context, reference=reference)
    if codec.encode_calls - before != len(clip):
        raise RuntimeError("EncodeOnce contract violated")
    return total, records


def legacy_frame(codec, encoded, objective, rgb, crops, budgets, context, prediction, inspect):
    measured = []
    for index, budget in enumerate(budgets):
        rec = render_training(codec, encoded.tokens, prediction, context, budget)
        measured.append(
            loss_record(objective, rgb, rec, crops, encoded.frame_id, budget, inspect and index == 0)
        )
    losses, records = zip(*measured)
    return torch.stack(losses).mean(), records
