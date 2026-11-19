"""Vectorized training prefixes with one receiver history context per frame.

Inference retains its O(delta-K) packet API. Training has all IDs already and
forms independent differentiable prefixes without repeatedly running the same
frozen history network or scattering old training tokens packet by packet.
"""

from dataclasses import dataclass

import torch

from aip_vtok.decoder.latent_canvas import LatentCanvas


def prefix_canvas(tokenizer, tokens, prediction, budget, differentiable=True):
    canvas = LatentCanvas(tokenizer, tokens.plan, prediction)
    values = tokens.codes if differentiable else tokenizer.fsq.ids_to_codes(tokens.ids)
    passes, positions = tokens.plan.pass_ids[:budget], tokens.plan.positions[:budget]
    for pass_id in range(3):
        selected = passes == pass_id
        canvas.buffers[pass_id] = tokenizer.synthesize_pass(
            values[:, :budget][:, selected], positions[selected], pass_id, tokens.plan.latent_shape
        )
        canvas.masks[pass_id].flatten()[positions[selected]] = True
    canvas.count = canvas.projected_tokens = budget
    return canvas


def receiver_context(codec, state, frame_id, tokens):
    context = codec.backbone.begin_frame(state, frame_id)
    prediction = codec.base_latent(state, context, tokens.plan.latent_shape)
    return context, prediction


def prefix_latents(codec, tokens, prediction, budget, differentiable=True):
    canvas = prefix_canvas(codec.tokenizer, tokens, prediction, budget, differentiable)
    raw = canvas.latent()
    return raw, codec.fusion(raw, canvas.masks)


def render_training(codec, tokens, prediction, context, budget):
    _, latent = prefix_latents(codec, tokens, prediction, budget)
    rec, _ = codec.backbone.decode_aligned(latent, context, checkpointed=True)
    return rec


@dataclass(frozen=True)
class TrainingReference:
    latent: torch.Tensor
    feature: torch.Tensor
    budget: int


def reference_prefix(codec, tokens, prediction, context, differentiable):
    budget = min(codec.reference_i if context.intra else codec.reference_p, len(tokens.plan.positions))
    latent = prefix_canvas(codec.tokenizer, tokens, prediction, budget, differentiable).latent()
    return TrainingReference(latent, codec.projector(latent, context.intra), budget)


def training_reference(codec, encoded):
    return reference_prefix(codec, encoded.tokens, encoded.prediction, encoded.context, differentiable=True)


@torch.no_grad()
def commit_training_reference(codec, tokens, prediction, context, *, reference=None):
    if reference is None:
        reference = reference_prefix(codec, tokens, prediction, context, differentiable=False)
    return context.commit_reference_once(reference.feature, reference.latent)
