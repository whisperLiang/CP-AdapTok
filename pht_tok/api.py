"""Encode and decode the next native coding unit on one CUDA device."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from baselines.dcvc_uf.native import NativeNetworks
from common.inputs import json_at
from common.paths import ROOT
from pht_tok.bitstream import Chunk, Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from third_party.DCVC.src.utils.transforms import rgb2ycbcr, ycbcr2rgb

NATIVE_FRAMES = {"ld": 1, "hts": 8}


@dataclass(frozen=True, slots=True)
class DecodeResult:
    frames: Tensor | None
    stage: int
    reference_committed: bool
    consumed_bytes: int


class PHTTok:
    """One I-frame followed by LD frames or HT-S chunks of up to eight RGB frames."""

    def __init__(self, backend: str, checkpoint: str | Path, device: str = "cuda", *,
                 layers: int | None = None, reference_layers: int | None = None) -> None:
        if backend not in NATIVE_FRAMES:
            raise ValueError("backend must be ld or hts")
        self.mode = self._mode(backend, layers)
        self.reference_layers = reference_layers if reference_layers is not None else {Mode.MASK: 2}.get(self.mode, 1)
        if not 1 <= self.reference_layers <= self.mode.layers:
            raise ValueError("reference_layers must select a legal prefix")
        placement = torch.device(device)
        if placement.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("the official inference backends require CUDA")
        self.stream = torch.cuda.Stream(device=placement)
        self.backend = backend
        checkpoint = Path(checkpoint)
        self.weights = checkpoint.parent if checkpoint.is_file() else checkpoint
        self.skip = official_settings()["skip_threshold"]
        self.model_id = model_identifier(self.weights, self.skip, self.backend)
        self.tx: PHTEncoder | None = None
        self.rx: PHTDecoder | None = None
        self._next_frame = 0

    @staticmethod
    def _mode(backend: str, layers: int | None) -> Mode:
        if backend == "ld" and layers in (None, 8):
            return Mode.LD_MASK
        if backend == "hts" and layers in (None, 4, 8):
            return Mode.MASK_FINE if layers == 8 else Mode.MASK
        raise ValueError("layer counts are 8 for LD and 4 or 8 for HT-S")

    @contextmanager
    def _device_stream(self):
        caller_stream = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller_stream)
        try:
            with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
                yield
        finally:
            caller_stream.wait_stream(self.stream)

    @torch.inference_mode()
    def encode(self, video_unit: Tensor, *, qp: int, intra: bool | None = None, reset: bool = False) -> Chunk:
        first = not self._next_frame if intra is None else intra
        if reset and first:
            raise ValueError("an independent I-frame already resets the reference")
        with self._device_stream():
            frames = centered_yuv(video_unit, 1 if first else NATIVE_FRAMES[self.backend], self.stream.device)
            if self.tx is None:
                self.tx = PHTEncoder(NativeNetworks(self.weights, self.backend, self.skip),
                                     self.model_id, backend=self.backend)
            if first:
                packet = self.tx.encode_intra(frames, qp, self._next_frame)
            else:
                packet = self.tx.encode_chunk_once(frames, qp, self._next_frame, self.reference_layers,
                                                   mode=self.mode, reset=reset)
        self._next_frame += len(video_unit)
        return packet

    @torch.inference_mode()
    def decode(self, payload: bytes, *, new_unit: bool = True, offset: int | None = None) -> DecodeResult:
        """Begin an access unit or append bytes at their original offset."""
        with self._device_stream():
            if self.rx is None:
                self.rx = PHTDecoder(NativeNetworks(self.weights, self.backend, self.skip),
                                     self.model_id, backend=self.backend)
            session = self.rx.begin_chunk() if new_unit else self.rx.session
            if session is None:
                raise ValueError("start receiving an access unit before appending a fragment")
            session.append_bytes(payload, offset=offset)
            session.decode_new_layers()
            output = ycbcr2rgb(torch.cat(session.reconstruct()) + 0.5).float() if session.stage else None
        return DecodeResult(output, session.stage, session.reference_committed,
                            session.parser.received - len(session.parser.buffer))


def official_settings():
    return json_at(ROOT / "configs/pht_tok/protocol.json")["official_settings"]


def centered_yuv(video: Tensor, maximum: int, device: torch.device) -> Tensor:
    if video.ndim != 4 or video.shape[1] != 3 or not 1 <= len(video) <= maximum:
        raise ValueError(f"expected 1–{maximum} native RGB frames with shape T,3,H,W")
    if not video.is_floating_point() and video.dtype != torch.uint8:
        raise ValueError("RGB input must be floating point in [0, 1] or unsigned eight-bit")
    rgb = video.to(device=device, dtype=torch.float32)
    if video.dtype == torch.uint8:
        rgb = rgb / 255
    return rgb2ycbcr(rgb).half() - 0.5
