"""Whole-clip audit built from the public frame-local T0 entry points."""

from hla_tok import Geometry


def reconstruct_clip(model, frames, budget):
    for frame in frames:
        ids = model.encode_frame(frame, budget)
        yield ids, model.decode_prefix(ids, Geometry(*frame.shape[-2:]))
