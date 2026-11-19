"""Measured loss and module gradients at named graph boundaries."""

import torch


def module_gradient_norm(module):
    gradients = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def gradient_diagnostics(codec):
    modules = {
        "anchor_encoder": codec.tokenizer.anchor,
        "innovation_encoder": codec.tokenizer.innovation,
        "detail_encoder": codec.tokenizer.detail,
        "fusion": codec.fusion,
    }
    if codec.predictor is not None:
        modules["temporal_predictor"] = codec.predictor
    else:
        modules["projector"] = codec.projector
    modules.update({f"lift{index}": module for index, module in enumerate(codec.tokenizer.lifts)})
    return {name: module_gradient_norm(module) for name, module in modules.items()}


def loss_record(objective, rgb, rec, crops, frame_id, budget, inspect):
    loss, components = objective(rgb, rec, crops)
    record = {
        "frame": frame_id,
        "budget": budget,
        "components": {name: float(value.detach()) for name, value in components.items()},
    }
    if inspect:
        record["weighted_loss_gradient_norm_at_rgb"] = {
            name: float(
                torch.autograd.grad(objective.weights[name] * value, rec, retain_graph=True)[0].norm()
            )
            for name, value in components.items()
        }
    return loss, record
