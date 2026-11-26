"""Use the official HT-S native analysis, spatial priors and synthesis."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from hla_tok.hts.native import tensor_digest

GROUPS = 4
CHUNK_FRAMES = 8


@dataclass(frozen=True, slots=True)
class Reference:
    intra: bool
    tensors: tuple[Tensor, ...]

    def digest(self) -> str:
        return f"{int(self.intra)}:{tensor_digest(list(self.tensors))}"


@dataclass(frozen=True, slots=True)
class EncodedLatents:
    """Borrowed buffers, valid until the adapter analyses its next chunk."""

    hyper: Tensor
    residuals: Tensor
    packed: Tensor
    keep: Tensor


class HTSAdapter:
    def __init__(self, model: Any):
        if not model.is_hts:
            raise ValueError("PHT-Tok requires the HT-S spatial prior")
        self.model = model
        self.reference: Reference | None = None

    @property
    def proxy(self) -> Any:
        return self.model.proxy

    def initialize(self, reconstructed_i: Tensor) -> None:
        """Install an independently decoded, padded official I-frame."""
        self.model.clear_dpb()
        self.model.add_ref_feature_from_frame(reconstructed_i, apply_feature_adaptor=False)
        if not hasattr(self.proxy, "pht_analyze"):
            raise RuntimeError("the pinned PHT CUDA extension is required")
        self.commit_reference(self.snapshot_reference(intra=True))

    def snapshot_reference(self, intra: bool) -> Reference:
        return Reference(intra, tuple(self.proxy.audit_reference(intra)))

    def commit_reference(self, reference: Reference) -> None:
        self.reference = reference

    def _prepare(self, qp: int) -> None:
        if not 0 <= qp <= 63:
            raise ValueError("QP must be in the official 0–63 range")
        if self.reference is None:
            raise RuntimeError("no committed I-frame or P-prefix reference")
        self.proxy.pht_load_reference(self.reference.tensors, self.reference.intra)
        self.proxy.pht_prepare()

    def encode_chunk(self, frames: Tensor, qp: int) -> EncodedLatents:
        if frames.ndim != 4 or frames.shape[:2] != (1, CHUNK_FRAMES * 3):
            raise ValueError("expected a single eight-frame YCbCr chunk")
        if not frames.is_cuda or frames.dtype != torch.float16:
            raise ValueError("the native HT-S adapter requires CUDA FP16")
        self._prepare(qp)
        padded_bottom, padded_right = (-frames.shape[2]) % 16, (-frames.shape[3]) % 16
        self.proxy.pht_analyze(frames, qp, padded_bottom, padded_right)
        return EncodedLatents(*self.proxy.pht_symbols(True))

    def start_decoding(self, hyper: Tensor, qp: int) -> tuple[Tensor, Tensor]:
        expected = self.proxy.pht_symbols(False)[0]
        if hyper.shape != expected.shape or hyper.dtype != expected.dtype or hyper.device != expected.device:
            raise ValueError("hyperprior must match the official tensor layout and integer type")
        self._prepare(qp)
        self.proxy.pht_prior(hyper, qp)
        _, _, indexes, keep = self.proxy.pht_symbols(False)
        return indexes, keep

    def masks(self) -> tuple[Tensor, ...]:
        return tuple(self.proxy.pht_masks())

    def apply_group(self, residuals: Tensor, group: int) -> None:
        if not 0 <= group < GROUPS:
            raise ValueError("unknown HT-S spatial group")
        self.proxy.pht_group(residuals, group)

    def reconstruct(self, groups: int, qp: int, *, reset: bool, display: bool = True) -> list[Tensor]:
        if not 1 <= groups <= GROUPS:
            raise ValueError("at least one complete group is required")
        return list(self.proxy.pht_render(groups, qp, display, reset))
