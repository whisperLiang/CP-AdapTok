"""One EncodeOnce and one receiver history context for all three training budgets."""

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.training.budgets import sample_budgets
from aip_vtok.training.diagnostics import loss_record
from aip_vtok.training.losses import crop_plan
from aip_vtok.training.prefix import commit_training_reference, receiver_context, render_training


def paired_step(codec, clip, objective, rng, inspect_gradients=False):
    state, total, records = ReferenceState(), 0.0, []
    before = codec.encode_calls
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        encoded = codec.encode_frame(rgb, state, frame_id)
        context, prediction = receiver_context(codec, state, frame_id, encoded.tokens)
        crops = crop_plan(1080, 1920, rng.crop_rng)
        losses = []
        for index, budget in enumerate(sample_budgets(rng)):
            rec = render_training(codec, encoded.tokens, prediction, context, budget)
            loss, record = loss_record(
                objective,
                rgb,
                rec,
                crops,
                frame_id,
                budget,
                inspect_gradients and frame_id == 0 and index == 0,
            )
            losses.append(loss)
            records.append(record)
        loss = torch.stack(losses).mean() / len(clip)
        loss.backward()
        total += float(loss.detach())
        state = commit_training_reference(codec, encoded.tokens, prediction, context)
    if codec.encode_calls - before != len(clip):
        raise RuntimeError("EncodeOnce contract violated")
    return total, records
