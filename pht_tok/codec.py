"""PHT one-pass encoder and independent incremental receiver."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from baselines.dcvc_uf.native import NativeNetworks, decode_intra
from baselines.dcvc_uf.official import NativeNAL
from pht_tok.bitstream import NO_REFERENCE, Chunk, ChunkHeader, CorruptStream, Mode, StreamReader
from pht_tok.entropy import NativeEntropy
from pht_tok.official_adapter import HTSBackend, LDBackend, NativeBackend, NativeSymbols
from pht_tok.official_adapter import assemble as assemble
from pht_tok.precision import WIDTHS


class MissingReference(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceKey:
    chunk_id: int
    version: int
    height: int
    width: int


def flat_nhwc(values: Tensor) -> Tensor:
    return values.permute(0, 2, 3, 1).reshape(-1)


class Layout:
    def __init__(self, adapter: NativeBackend):
        masks = adapter.masks()
        _, channels, h, w = masks[0].shape
        self.shape = (1, h, w, channels)
        self.device = masks[0].device
        self.positions = tuple(np.flatnonzero(flat_nhwc(mask).cpu().numpy()) for mask in masks)
        self.size = masks[0].numel()

    def active(self, keep: np.ndarray, layers: int | None = None) -> tuple[np.ndarray, ...]:
        partitions = (layers or len(self.positions)) // len(self.positions)
        return tuple(part[keep[part]] for indices in self.positions
                     for part in (indices[i::partitions] for i in range(partitions)))

    def to_spatial(self, dense: Tensor) -> Tensor:
        return dense.view(self.shape).permute(0, 3, 1, 2)

    def zeros(self) -> Tensor:
        return torch.zeros(self.size, device=self.device, dtype=torch.float16)

    def scatter(self, target: Tensor, positions: np.ndarray, values: np.ndarray) -> None:
        indexes = torch.from_numpy(positions).to(self.device)
        symbols = torch.from_numpy(values).to(device=self.device, dtype=torch.float16)
        target.index_copy_(0, indexes, symbols)


def model_identifier(weights: Path, skip: float, backend: str = "hts") -> bytes:
    variant = b"PHT1/HTS" if backend == "hts" else b"PHT2/LD"
    digest = sha256(variant + b"/cbdae87a/" + str(skip).encode())
    for name in ("cvpr2026_image.pth.tar", f"cvpr2026_video_{backend}.pth.tar"):
        with (weights / name).open("rb") as handle:
            for part in iter(lambda: handle.read(2**20), b""):
                digest.update(part)
    return digest.digest()[:8]


class Peer:
    def __init__(self, networks: NativeNetworks, model_id: bytes, *, backend: str = "hts"):
        self.adapter: NativeBackend = {"hts": HTSBackend, "ld": LDBackend}[backend](networks.video)
        self.backend = backend
        self.entropy = NativeEntropy(networks.video)
        self.intra = networks.intra
        self.model_id = model_id
        self.key: ReferenceKey | None = None
        self.layout: Layout | None = None

    def install(self, header: ChunkHeader, reconstructed: Tensor) -> None:
        if self.key is None or (self.key.height, self.key.width) != (header.height, header.width):
            self.layout = None
        self.adapter.initialize(reconstructed)
        self.key = ReferenceKey(header.chunk_id, header.reference_version, header.height, header.width)
        if self.layout is None:
            self.layout = Layout(self.adapter)

    def commit(self, header: ChunkHeader) -> None:
        self.adapter.commit_reference(self.adapter.snapshot_reference(header.reset_memory))
        self.key = ReferenceKey(header.chunk_id, header.reference_version, header.height, header.width)


class PHTEncoder(Peer):
    @torch.inference_mode()
    def encode_intra(self, frame: Tensor, qp: int, chunk_id: int) -> Chunk:
        if frame.ndim != 4 or frame.shape[:2] != (1, 3) or frame.dtype != torch.float16 or not frame.is_cuda:
            raise ValueError("expected one centered CUDA FP16 YCbCr frame")
        height, width = frame.shape[-2:]
        if self.key is not None and chunk_id <= self.key.chunk_id:
            raise ValueError("I-frame IDs must advance")
        version = 0 if self.key is None else self.key.version + 1
        mode = Mode.INTRA if self.backend == "hts" else Mode.LD_INTRA
        header = ChunkHeader(self.model_id, chunk_id, NO_REFERENCE, version, height, width,
                             qp, 1, mode, 1, False)
        encoded = self.intra.compress(frame.contiguous(memory_format=torch.channels_last), qp,
                                      (-height) % 16, (-width) % 16)
        info = FrameInfo(chunk_id, height, width, qp, True, encoded["ec_parallel"], False)
        nal = NativeNAL().encode(BaseFrame(info, encoded["bit_stream"]))
        stream = Chunk.pack(header, (nal,))
        self.install(header, encoded["x_hat"])
        return stream

    @torch.inference_mode()
    def encode_chunk_once(self, frames: Tensor, qp: int, chunk_id: int, k_ref: int,
                          *, mode: Mode | None = None, reset: bool = False, commit: bool = True) -> Chunk:
        if self.key is None or self.layout is None:
            raise MissingReference("send a complete I-frame first")
        if mode is None:
            mode = Mode.MASK if self.backend == "hts" else Mode.LD_MASK
        if frames.shape[-2:] != (self.key.height, self.key.width) or mode.is_intra or mode.backend != self.backend:
            raise ValueError("a progressive chunk must match its reference dimensions")
        header = ChunkHeader(self.model_id, chunk_id, self.key.chunk_id, self.key.version,
                             self.key.height, self.key.width, qp, len(frames), mode, k_ref, reset)
        analysis = self.adapter.analyze(frames, qp)
        payloads = (self._precision_payloads(analysis, header, self.layout, commit) if mode == Mode.PRECISION else
                    self._mask_payloads(analysis, header, self.layout, commit))
        stream = Chunk.pack(header, payloads)
        if commit:
            self.commit(header)
        return stream


    def _mask_payloads(self, analysis: NativeSymbols, header: ChunkHeader, layout: Layout,
                       commit: bool) -> tuple[bytes, ...]:
        hyper = flat_nhwc(analysis.hyper).cpu().numpy()
        packed = analysis.packed.cpu().numpy()
        keep = analysis.keep.cpu().numpy()
        positions = layout.active(keep, header.mode.layers)
        grouped = tuple(packed[indices] for indices in positions)
        self.entropy.start_mask_encoding(hyper, header.qp, analysis.hyper.shape[1], grouped)
        if commit:
            residuals = analysis.residuals
            if header.mode != Mode.MASK and header.k_ref < header.mode.layers:
                dense = layout.zeros()
                selected = np.concatenate(positions[:header.k_ref])
                layout.scatter(dense, selected, (packed[selected] >> 8).astype(np.int8))
                residuals = layout.to_spatial(dense)
            self.adapter.render(header, header.k_ref, residuals, display=False, received=False)
        return self.entropy.finish_mask_encoding()

    def _precision_payloads(self, analysis: NativeSymbols, header: ChunkHeader, layout: Layout,
                            commit: bool) -> tuple[bytes, ...]:
        hyper = flat_nhwc(analysis.hyper).cpu().numpy()
        keep = analysis.keep.cpu().numpy()
        z = self.entropy.encode_z(hyper, header.qp, analysis.hyper.shape[1])
        layers, preview = encode_precision(self.entropy, analysis.packed.cpu().numpy(), keep, header.k_ref)
        if commit:
            dense = layout.zeros()
            layout.scatter(dense, np.flatnonzero(keep), preview)
            self.adapter.render(header, header.k_ref, layout.to_spatial(dense), display=False, received=False)
        return (z, *layers)


def encode_precision(entropy: NativeEntropy, packed: np.ndarray, keep: np.ndarray,
                     k_ref: int) -> tuple[tuple[bytes, ...], np.ndarray]:
    active = packed[keep]
    indexes = (active & 255).astype(np.uint8)
    values = (active >> 8).astype(np.int8)
    layers = entropy.precision.encode(indexes, values)
    parents = (values.astype(np.int16) + 128) // WIDTHS[k_ref - 1]
    return layers, entropy.precision.preview(parents, k_ref - 1)


class ProgressiveDecodeSession:
    def __init__(self, peer: "PHTDecoder", commit: bool):
        self.peer = peer
        self.parser = StreamReader()
        self.pending: list[bytes] = []
        self.stage = 0
        self.entropy_decodes = 0
        self.display_reconstructions = 0
        self.reference_committed = False
        self._allow_commit = commit
        self._ready = False
        self._parents: np.ndarray | None = None
        self._cache: list[Tensor] | None = None
        self._feature_stage = 0
        self._failed = False

    @property
    def header(self) -> ChunkHeader:
        if self.parser.header is None:
            raise ValueError("the full header is required")
        return self.parser.header

    def _active(self) -> None:
        if self.peer.session is not self or self._failed:
            raise ValueError("chunk session is closed")

    def append_bytes(self, data: bytes, *, offset: int | None = None) -> None:
        self._active()
        self.pending.extend(self.parser.append_bytes(data, offset=offset))

    @torch.inference_mode()
    def decode_new_layers(self) -> int:
        self._active()
        try:
            for payload in self.pending:
                self._decode(payload)
        except (ValueError, RuntimeError):
            self._failed = True
            raise
        self.pending.clear()
        return self.stage

    def _decode(self, payload: bytes) -> None:
        header = self.header
        if not self._ready:
            self.peer.check_header(header)
            self._ready = True
            if header.mode.is_intra:
                self._cache = [self.peer.decode_intra(header, payload)]
                self.stage, self.reference_committed = 1, True
            else:
                self._decode_hyper(payload)
            self.entropy_decodes += 1
            return
        self._decode_enhancement(payload)
        self.entropy_decodes += 1
        self.stage += 1
        self._cache = None
        if self.stage == header.k_ref and self._allow_commit:
            self.commit_reference()

    def _decode_hyper(self, payload: bytes) -> None:
        header, peer = self.header, self.peer
        template = peer.adapter.proxy.pht_symbols(False)[0]
        hyper = peer.entropy.decode_z(payload, header.qp, tuple(template.shape), template.device)
        indexes, keep = peer.adapter.start_decoding(hyper, header.qp)
        self._indexes = indexes.cpu().numpy()
        layout = peer.layout
        assert layout is not None
        enabled = keep.cpu().numpy()
        self._positions = ((np.flatnonzero(enabled),) if header.mode == Mode.PRECISION
                           else layout.active(enabled, header.mode.layers))
        self._dense = layout.zeros()

    def _decode_enhancement(self, payload: bytes) -> None:
        peer = self.peer
        layout = peer.layout
        assert layout is not None
        if self.header.mode == Mode.PRECISION:
            positions = self._positions[0]
            self._parents = peer.entropy.precision.decode(
                payload, self._indexes[positions], self.stage, self._parents
            )
            layout.scatter(self._dense, positions, peer.entropy.precision.preview(self._parents, self.stage))
        else:
            positions = self._positions[self.stage]
            values = peer.entropy.decode_y(payload, self._indexes[positions])
            layout.scatter(self._dense, positions, values)
            peer.adapter.accept_mask_layer(self.header, self.stage, layout.to_spatial(self._dense))

    def _render(self, display: bool) -> list[Tensor]:
        out = self.peer.adapter.render(self.header, self.stage, self.residuals(), display=display, received=True)
        self._feature_stage = self.stage
        return out

    @torch.inference_mode()
    def commit_reference(self) -> None:
        self._active()
        if self.reference_committed:
            return
        if self.stage != self.header.k_ref:
            raise MissingReference("the negotiated reference prefix has not arrived")
        if self._feature_stage != self.stage:
            self._render(display=False)
        self.peer.commit(self.header)
        self.reference_committed = True

    @torch.inference_mode()
    def reconstruct(self) -> list[Tensor]:
        self._active()
        if not self.stage:
            raise MissingReference("no complete display prefix")
        if self._cache is None:
            h = self.header
            frames = (self.peer.adapter.display_feature() if self._feature_stage == self.stage
                      else self._render(display=True))
            self.display_reconstructions += 1
            self._cache = [frame[..., :h.height, :h.width].clone() for frame in frames[:h.valid_frames]]
        return self._cache

    def residuals(self) -> Tensor:
        assert self.peer.layout is not None
        return self.peer.layout.to_spatial(self._dense)


class PHTDecoder(Peer):
    session: ProgressiveDecodeSession | None = None

    def begin_chunk(self, *, commit: bool = True) -> ProgressiveDecodeSession:
        self.session = ProgressiveDecodeSession(self, commit)
        return self.session

    def decode_prefix(self, prefix: bytes, *, commit: bool = True) -> ProgressiveDecodeSession:
        session = self.begin_chunk(commit=commit)
        session.append_bytes(prefix)
        session.decode_new_layers()
        return session

    def check_header(self, header: ChunkHeader) -> None:
        if header.mode.backend != self.backend:
            raise CorruptStream("PHT backend mismatch")
        if header.model != self.model_id:
            raise CorruptStream("PHT model/configuration identity mismatch")
        if header.mode.is_intra:
            if self.key is not None and (header.chunk_id <= self.key.chunk_id
                                         or header.reference_version <= self.key.version):
                raise MissingReference("stale refresh")
        elif self.key != ReferenceKey(header.parent_id, header.reference_version, header.height, header.width):
            raise MissingReference("required reference chunk or version is missing")

    def decode_intra(self, header: ChunkHeader, nal: bytes) -> Tensor:
        packet = NativeNAL().decode(nal, header.chunk_id)
        info = packet.info
        if not info.intra or (info.height, info.width, info.qp) != (header.height, header.width, header.qp):
            raise CorruptStream("PHT refresh and official I-frame metadata disagree")
        frames = decode_intra(self.intra, packet)
        self.install(header, frames)
        return frames[..., :info.height, :info.width].clone()
