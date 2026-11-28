"""Use the official HT-S native analysis, spatial priors and synthesis."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from common.integrity import tensor_digest
from pht_tok.bitstream import ChunkHeader, Mode

GROUPS = 4
CHUNK_FRAMES = 8


@dataclass(frozen=True, slots=True)
class Reference:
    intra: bool
    tensors: tuple[Tensor, ...]
    _versions: tuple[int | None, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_versions", tuple(None if t.is_inference() else t._version for t in self.tensors))

    def digest(self) -> str:
        return f"{int(self.intra)}:{tensor_digest(list(self.tensors))}"

    def require_intact(self) -> None:
        if any(version is not None and tensor._version != version
               for tensor, version in zip(self.tensors, self._versions, strict=True)):
            raise ValueError("committed reference tensors were modified")


@dataclass(frozen=True, slots=True)
class NativeSymbols:
    """Borrowed buffers from the official analysis and entropy prior."""

    hyper: Tensor
    residuals: Tensor
    packed: Tensor
    keep: Tensor


EncodedLatents = NativeSymbols


def require_centered_frames(frames: Tensor, maximum: int) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3 or not 1 <= len(frames) <= maximum:
        raise ValueError(f"provide 1–{maximum} consecutive native YCbCr frames")
    if frames.dtype != torch.float16 or not frames.is_cuda:
        raise ValueError("provide centered CUDA FP16 YCbCr frames")


def assemble(frames: Tensor) -> Tensor:
    require_centered_frames(frames, CHUNK_FRAMES)
    padding = frames[-1:].expand(CHUNK_FRAMES - len(frames), -1, -1, -1)
    return torch.cat((frames, padding)).reshape(1, 24, *frames.shape[-2:]).contiguous(
        memory_format=torch.channels_last
    )


class NativeBackend:
    def __init__(self, model: Any):
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
            raise RuntimeError("the pinned PHT CUDA extension is required for this backend")
        self.commit_reference(self.snapshot_reference(intra=True))

    def snapshot_reference(self, intra: bool) -> Reference:
        # Keep native snapshots versioned even when the caller uses inference_mode.
        with torch.inference_mode(False):
            return Reference(intra, tuple(self.proxy.audit_reference(intra)))

    def commit_reference(self, reference: Reference) -> None:
        self.reference = reference

    def _prepare(self, qp: int) -> None:
        if not 0 <= qp <= 63:
            raise ValueError("QP must be in the official 0–63 range")
        if self.reference is None:
            raise RuntimeError("no committed I-frame or P-prefix reference")
        self.reference.require_intact()
        self.proxy.pht_load_reference(self.reference.tensors, self.reference.intra)
        self.proxy.pht_prepare()

    def encode_chunk(self, frames: Tensor, qp: int) -> NativeSymbols:
        if not frames.is_cuda or frames.dtype != torch.float16:
            raise ValueError("native analysis requires CUDA FP16")
        self._prepare(qp)
        padded_bottom, padded_right = (-frames.shape[2]) % 16, (-frames.shape[3]) % 16
        self.proxy.pht_analyze(frames, qp, padded_bottom, padded_right)
        return NativeSymbols(*self.proxy.pht_symbols(True))

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

    def analyze(self, frames: Tensor, qp: int) -> NativeSymbols:
        return self.encode_chunk(frames, qp)

    def accept_mask_layer(self, header: ChunkHeader, stage: int, residuals: Tensor) -> None:
        """Backends may retain a native group's synthesis inputs between layers."""

    def render(self, header: ChunkHeader, stage: int, residuals: Tensor, *, display: bool,
               received: bool) -> list[Tensor]:
        raise NotImplementedError

    def display_feature(self) -> list[Tensor]:
        raise NotImplementedError


class HTSBackend(NativeBackend):
    def __init__(self, model: Any):
        if not model.is_hts:
            raise ValueError("PHT-Tok requires the HT-S spatial prior")
        super().__init__(model)

    def analyze(self, frames: Tensor, qp: int) -> NativeSymbols:
        return self.encode_chunk(assemble(frames), qp)

    def encode_chunk(self, frames: Tensor, qp: int) -> NativeSymbols:
        if frames.ndim != 4 or frames.shape[:2] != (1, CHUNK_FRAMES * 3):
            raise ValueError("expected a single eight-frame YCbCr chunk")
        return super().encode_chunk(frames, qp)

    def apply_group(self, residuals: Tensor, group: int) -> None:
        if not 0 <= group < GROUPS:
            raise ValueError("unknown HT-S spatial group")
        self.proxy.pht_group(residuals, group)

    def reconstruct(self, groups: int, qp: int, *, reset: bool, display: bool = True) -> list[Tensor]:
        if not 1 <= groups <= GROUPS:
            raise ValueError("at least one complete group is required")
        return list(self.proxy.pht_render(groups, qp, display, reset))

    def display_feature(self) -> list[Tensor]:
        return list(self.proxy.pht_display())

    def accept_mask_layer(self, header: ChunkHeader, stage: int, residuals: Tensor) -> None:
        if header.mode == Mode.MASK:
            self.apply_group(residuals, stage)

    def render(self, header: ChunkHeader, stage: int, residuals: Tensor, *, display: bool,
               received: bool) -> list[Tensor]:
        if header.mode == Mode.PRECISION:
            groups = GROUPS
        else:
            partitions = header.mode.layers // GROUPS
            groups = (stage + partitions - 1) // partitions
        if not received or header.mode != Mode.MASK:
            for group in range(groups):
                self.apply_group(residuals, group)
        return self.reconstruct(groups, header.qp, reset=header.reset_memory, display=display)


class LDBackend(NativeBackend):
    def encode_chunk(self, frames: Tensor, qp: int) -> NativeSymbols:
        if frames.ndim != 4 or frames.shape[:2] != (1, 3):
            raise ValueError("LD requires one native frame")
        return super().encode_chunk(frames.contiguous(memory_format=torch.channels_last), qp)

    def render(self, header: ChunkHeader, stage: int, residuals: Tensor, *, display: bool,
               received: bool) -> list[Tensor]:
        self.proxy.pht_render(residuals, header.qp, header.reset_memory and not display)
        return self.display_feature() if display else []

    def display_feature(self) -> list[Tensor]:
        return [self.proxy.pht_display()]


HTSAdapter = HTSBackend
