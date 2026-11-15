import hashlib
import struct
from dataclasses import dataclass, field, replace

import torch


def tensor_digest(tensor):
    """Device-side, integer-only non-authenticating fingerprint; transfer just 16 bytes.

    Mix float32 bit patterns with position, using modulo-2^64 integer arithmetic.
    Integer reductions are order independent, unlike floating summary statistics.
    This detects accidental drift; CRC/fingerprints are not cryptographic authentication.
    """
    words = tensor.detach().float().contiguous().view(torch.int32).flatten()
    summary = fingerprint_summary(words)
    raw = summary.cpu().numpy().astype(">i8", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()[:32]


def fingerprint_summary(words):
    positions = torch.arange(1, words.numel() + 1, device=words.device, dtype=torch.int64)
    return _mix(words, positions)


def mix_fingerprint_words(words, positions):
    words = words.to(torch.int64)
    mixed = words ^ (positions * -7046029254386353131)
    mixed = (mixed ^ ((mixed >> 30) & ((1 << 34) - 1))) * -4658895280553007687
    mixed = (mixed ^ ((mixed >> 27) & ((1 << 37) - 1))) * -7723592293110705685
    mixed = mixed ^ ((mixed >> 31) & ((1 << 33) - 1))
    return torch.stack((mixed.sum(), (mixed * (positions * 2 + 1)).sum()))


_mix = mix_fingerprint_words


def configure_fingerprint_backend(name="eager"):
    """One pure integer algorithm; optional fusion never changes its hash semantics."""
    global _mix
    if name == "eager":
        _mix = mix_fingerprint_words
    elif name == "compiled":
        _mix = torch.compile(mix_fingerprint_words, fullgraph=True, dynamic=True)
    else:
        raise ValueError("unknown integer fingerprint backend")


def state_versions(tensors):
    if any(tensor is not None and tensor.is_inference() for tensor in tensors):
        return None
    return tuple(None if tensor is None else tensor._version for tensor in tensors)


@dataclass(frozen=True)
class CausalState:
    model_id: str
    frame_id: int = -1
    feature: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    base_digest: str = ""
    _cached_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)
    _cached_versions: tuple | None = field(default=(), init=False, repr=False, compare=False)

    @classmethod
    def empty(cls, model_id):
        return cls(model_id)

    def fingerprint(self):
        if self.frame_id == -1:
            return "00" * 16
        versions = state_versions((self.feature, self.memory))
        if (
            versions is not None
            and self._cached_fingerprint is not None
            and self._cached_versions == versions
        ):
            return self._cached_fingerprint
        digest = hashlib.sha256(
            self.model_id.encode() + struct.pack(">q", self.frame_id) + self.base_digest.encode()
        )
        for tensor in (self.feature, self.memory):
            if tensor is None:
                digest.update(b"none")
            else:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(bytes.fromhex(tensor_digest(tensor)))
        result = digest.hexdigest()[:32]
        object.__setattr__(self, "_cached_fingerprint", result)
        object.__setattr__(self, "_cached_versions", versions)
        return result

    def detach(self):
        detached = replace(
            self,
            feature=None if self.feature is None else self.feature.detach(),
            memory=None if self.memory is None else self.memory.detach(),
        )
        versions = state_versions((self.feature, self.memory))
        if versions is not None and self._cached_versions == versions:
            object.__setattr__(detached, "_cached_fingerprint", self._cached_fingerprint)
            object.__setattr__(detached, "_cached_versions", self._cached_versions)
        return detached


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
