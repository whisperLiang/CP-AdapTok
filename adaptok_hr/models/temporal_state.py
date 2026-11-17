"""Independent closed-loop state. No original RGB history is stored."""

import hashlib
from dataclasses import dataclass, replace

import torch


def stream_identity(name):
    return hashlib.sha256(name.encode("utf-8")).digest()[:16]


@dataclass(frozen=True)
class TemporalState:
    stream_id: bytes
    frame_index: int = -1
    frame_type: str = "I"
    gop: int = 32
    rgb: torch.Tensor | None = None
    feature: torch.Tensor | None = None
    version: int = 1
    chain: bytes = bytes(32)

    @property
    def next_index(self):
        return self.frame_index + 1

    @property
    def gop_position(self):
        return self.frame_index % self.gop

    def detach(self):
        return replace(self, rgb=self.rgb.detach(), feature=self.feature.detach())


def frame_context(state, gop):
    index = 0 if state is None else state.next_index
    is_intra = index % gop == 0
    return index, "I" if is_intra else "P", None if is_intra else state
