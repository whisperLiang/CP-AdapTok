"""Vectorized training prefixes with one receiver history context per frame.

Inference retains its O(delta-K) packet API. Training has all IDs already and
forms independent differentiable prefixes without repeatedly running the same
frozen history network or scattering old training tokens packet by packet.
"""

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
    device = codec.tokenizer.lifts[0].weight.device
    prediction = codec.predictor(state.latent, context.intra, tokens.plan.latent_shape, device)
    return context, prediction


def render_training(codec, tokens, prediction, context, budget):
    canvas = prefix_canvas(codec.tokenizer, tokens, prediction, budget)
    latent = codec.fusion(canvas.latent(), canvas.masks)
    rec, _ = codec.backbone.decode_aligned(latent, context, checkpointed=True)
    return rec


@torch.no_grad()
def commit_training_reference(codec, tokens, prediction, context):
    budget = codec.reference_i if context.intra else codec.reference_p
    canvas = prefix_canvas(codec.tokenizer, tokens, prediction, budget, differentiable=False)
    latent = canvas.latent()
    feature = codec.projector(latent, context.intra)
    return context.commit_reference_once(feature, latent)
