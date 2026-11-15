import hashlib
import struct
from dataclasses import dataclass, replace

import torch


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:32]


@dataclass(frozen=True)
class CausalState:
    model_id: str
    frame_id: int = -1
    feature: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    base_digest: str = ""

    @classmethod
    def empty(cls, model_id):
        return cls(model_id)

    def fingerprint(self):
        if self.frame_id == -1:
            return "00" * 16
        digest = hashlib.sha256(
            self.model_id.encode() + struct.pack(">q", self.frame_id) + self.base_digest.encode()
        )
        for tensor in (self.feature, self.memory):
            if tensor is None:
                digest.update(b"none")
            else:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(bytes.fromhex(tensor_digest(tensor)))
        return digest.hexdigest()[:32]

    def detach(self):
        return replace(
            self,
            feature=None if self.feature is None else self.feature.detach(),
            memory=None if self.memory is None else self.memory.detach(),
        )


@dataclass
class FrozenFrameContext:
    model_id: str
    frame_id: int
    frame_type: str
    height: int
    width: int
    qp: int
    reference_id: int
    reference_fingerprint: str
    context: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    committed: bool = False

    @property
    def latent_shape(self):
        return (self.height + 63) // 64 * 4, (self.width + 63) // 64 * 4

    def commit(self, feature, base_digest):
        if self.committed:
            raise RuntimeError("base state already committed for this frame")
        self.committed = True
        return CausalState(
            self.model_id,
            self.frame_id,
            feature,
            self.memory if self.frame_type == "P" else None,
            base_digest,
        )
