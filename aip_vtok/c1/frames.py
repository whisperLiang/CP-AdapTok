"""One immutable encoder context and the two coordinate systems per frame."""

from dataclasses import dataclass

import torch

from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.diagnostics.diag0.matched import CLEAN, LOW, low_frames
from aip_vtok.evaluation.fix import rgb_frame


@dataclass(frozen=True)
class Sample:
    context: FrameContext
    rgb: torch.Tensor
    native: torch.Tensor
    aligned: torch.Tensor

    @property
    def frame_type(self):
        return "I" if self.context.intra else "P"


@torch.no_grad()
def capture(backbone, state, frame_id, image):
    context = backbone.begin_frame(state, frame_id)
    rgb = rgb_frame(image)
    native = backbone.native_latent(rgb, context)
    return Sample(context, rgb, native, backbone.alignment.align(native, context.intra))


@torch.no_grad()
def clean_samples(clip, selected, backbone):
    state = ReferenceState()
    for index, image in enumerate(clip):
        sample = capture(backbone, state, index, image)
        _, feature = backbone.decode_aligned(sample.aligned, sample.context)
        state = sample.context.commit_reference_once(feature)
        if index in selected:
            yield sample


@torch.no_grad()
def history_samples(clip, selected, video, codec):
    backbone = codec.backbone
    for sample in clean_samples(clip, selected, backbone):
        yield CLEAN, sample
    for frame in low_frames(clip, selected, video, codec):
        native = backbone.native_latent(frame.rgb, frame.context)
        yield LOW, Sample(frame.context, frame.rgb, native, frame.y)


def frame_fields(sample, video, history, split):
    return {"video": video, "frame": sample.context.frame_id, "history": history,
            "frame_type": sample.frame_type, "split": split}
