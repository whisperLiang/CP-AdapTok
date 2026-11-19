"""One writer and an explicit commit lock, without a hidden RGB reference chain."""

import hashlib
from dataclasses import dataclass

import torch


def owned_tensor(tensor):
    return None if tensor is None else tensor.detach().clone()


@dataclass(frozen=True)
class ReferenceState:
    frame_id: int = -1
    feature: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    latent: torch.Tensor | None = None

    def is_intra(self, frame_id, gop):
        if frame_id != self.frame_id + 1:
            raise ValueError("reference frame IDs must be consecutive")
        return frame_id % gop == 0

    def fingerprint(self):
        digest = hashlib.sha256(str(self.frame_id).encode())
        for tensor in (self.feature, self.memory, self.latent):
            if tensor is not None:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()


@dataclass
class FrameContext:
    frame_id: int
    intra: bool
    context: torch.Tensor | None
    memory: torch.Tensor | None
    committed: bool = False

    def commit_reference_once(self, feature, latent=None):
        if self.committed:
            raise RuntimeError("reference already committed for this frame")
        channels = 192 if self.intra else 256
        if feature.ndim != 4 or feature.shape[:2] != (1, channels):
            raise ValueError("reference feature has incompatible native frame-type channels")
        if not self.intra and self.memory is None:
            raise ValueError("P reference requires synchronized memory")
        self.committed = True
        return ReferenceState(
            self.frame_id,
            owned_tensor(feature),
            None if self.intra else owned_tensor(self.memory),
            owned_tensor(latent),
        )
