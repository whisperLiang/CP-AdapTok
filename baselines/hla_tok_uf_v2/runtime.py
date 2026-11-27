"""Single native analysis, physical prefix families, and independent D0/D1/Anchor reception."""

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import torch
from torch import Tensor

from baselines.dcvc_uf.frames import BaseFrame
from baselines.dcvc_uf.official import NativeDecoder, NativeEncoder, OfficialAssets
from baselines.hla_tok_uf_v2.decoder import SharedDecoder, SharedRefinement
from baselines.hla_tok_uf_v2.entropy import EntropyTables
from baselines.hla_tok_uf_v2.latent import (
    LATENT_UNIT,
    MAX_APPLICATION_BYTES,
    Segmentation,
    UnifiedRefinement,
    normalized_error,
    stream_tag,
)
from baselines.hla_tok_uf_v2.native_weights import NativeAdjustments, install_adjustments
from baselines.hla_tok_uf_v2.records import BatchReceiver, BatchRecords, Display, quantized_symbols
from baselines.hla_tok_uf_v2.wire import APP_BASE_OVERHEAD, BaseBudgetExceeded, PrefixFamily, WireError
from common.video import NATIVE_SHAPE, restore_rgb

BASE_STANDARD_QP = (0, 8)
BASE_STANDARD_CAP = 9212


def check_base(frame: BaseFrame, decoder: NativeDecoder, cap: int) -> None:
    size = APP_BASE_OVERHEAD + len(frame.payload)
    if frame.info.dimensions != NATIVE_SHAPE:
        raise WireError("UF-V2 expects native 1920×1080 access units")
    if size > cap:
        decoder.order.desync = True
        raise BaseBudgetExceeded(size, cap)


@dataclass(frozen=True)
class Encoded:
    family: PrefixFamily
    reference: Tensor
    phase_ms: dict[str, float]



class Analysis(NativeEncoder):
    def proxy(self, intra: bool):
        return (self.intra if intra else self._video()).proxy

    def preview(self, base: BaseFrame) -> Tensor:
        pixels = self.proxy(base.info.intra).ufv2_preview()
        return restore_rgb(pixels.float(), *base.info.dimensions)

    def encode_base(self, rgb: Tensor, index: int, qp: tuple[int, int], intra: bool, reset: bool,
                    base_cap: int) -> BaseFrame:
        if rgb.shape[-2:] != NATIVE_SHAPE:
            raise WireError("UF-V2 expects native 1920×1080 input")
        record = self.encode(rgb, index, qp[0 if intra else 1], intra, reset)
        size = APP_BASE_OVERHEAD + len(record.payload)
        if size > base_cap:
            self.order.desync = True
            raise BaseBudgetExceeded(size, base_cap)
        return record




class Unified:
    def __init__(self, assets: OfficialAssets, model: UnifiedRefinement, qp: tuple[int, int],
                 refiner: SharedRefinement, bank: BatchRecords | None = None,
                 segmentation: Segmentation = Segmentation(8, 8), base_cap: int = BASE_STANDARD_CAP, family_id: str = "standard",
                 adjustments: NativeAdjustments | None = None):
        self.assets, self.native, self.qp, self.base_cap = assets, Analysis(assets), qp, base_cap
        self.model, self.refiner, self.adjustments = model, refiner, adjustments
        self._install(self.native)
        self.steps = {intra: model.quantizer(intra).step().detach() for intra in (True, False)}
        scales = tuple(model.scales().detach().cpu().tolist())
        legacy_cdf = EntropyTables.from_scales(scales).cdf
        legacy_tag = stream_tag(model, f"uf-latent-v1/{segmentation}/{legacy_cdf}")
        self.bank = bank or BatchRecords(scales, segmentation)
        self.tag = self._tag(legacy_tag, family_id)

    def _tag(self, legacy_tag: bytes, family_id: str) -> bytes:
        weights = self.bank.sensitivity
        digest = sha256(b"".join(weights[i].astype("<f8").tobytes() for i in (True, False))).hexdigest() if (
            weights is not None) else None
        suffix = f"/{self.adjustments.digest()}" if self.adjustments is not None else ""
        return stream_tag(self.refiner, f"uf-v2/{family_id}/{self.qp}/{self.base_cap}/{legacy_tag.hex()}/"
                                 f"{self.bank.segmentation}/{self.bank.static}/{self.bank.raw}/{digest}{suffix}")

    def _install(self, endpoint: NativeDecoder | NativeEncoder) -> None:
        if self.adjustments is not None:
            install_adjustments(endpoint, self.adjustments, self.qp[1])

    def reset(self) -> None:
        self.native.reset()

    def new_decoder(self) -> NativeDecoder:
        endpoint = SharedDecoder(self.assets)
        self._install(endpoint)
        return endpoint

    def receiver(self, decoder: NativeDecoder) -> BatchReceiver:
        return BatchReceiver(self.tag, lambda frame: self.prepare(decoder, frame))

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: BatchReceiver) -> bool:
        return torch.equal(encoded.reference, decoder.decoded_latent())

    def _display(self, decoder: NativeDecoder, frame: BaseFrame) -> Display:
        if not isinstance(decoder, SharedDecoder):
            raise TypeError("D1 requires its own independent shared-feature decoder")
        snapshot = decoder.shared_base(frame)
        return self._make_display(frame, snapshot.rgb, snapshot.scale,
                                  lambda delta: decoder.display(snapshot, self.refiner, delta))

    def _make_display(self, frame, base, scale, display) -> Display:
        info = frame.info
        steps = self.steps[info.intra]
        padded = tuple((size + post) // 16 for size, post in zip(info.dimensions, info.native_padding, strict=True))
        canvas = self.bank.canvas((steps.shape[1], *padded), info.intra)
        return Display(base, canvas, scale.float() * steps * LATENT_UNIT, display)

    @torch.no_grad()
    def prepare(self, decoder: NativeDecoder, frame: BaseFrame) -> Display:
        check_base(frame, decoder, self.base_cap)
        if frame.info.qp != self.qp[0 if frame.info.intra else 1]:
            raise WireError("base QP differs from prefix family")
        return self._display(decoder, frame)

    @torch.no_grad()
    def encode(self, rgb: Tensor, index: int, intra: bool, reset: bool = False,
               cap: int = MAX_APPLICATION_BYTES) -> Encoded:
        start = perf_counter()
        base = self.native.encode_base(rgb, index, self.qp, intra, reset, min(cap, self.base_cap))
        compressed = perf_counter()
        original, latent, scale = self.native.proxy(intra).ufv2_encode_snapshot()
        symbols = quantized_symbols(normalized_error(original, latent, scale), self.steps[intra])
        copied = perf_counter()
        family = self.bank.pack_symbols(base, self.tag, symbols, cap)
        end = perf_counter()
        return Encoded(family, latent, {"native_encode_ms": 1000 * (compressed - start),
                       "analysis_copy_ms": 1000 * (copied - compressed),
                       "group_rank_pack_ms": 1000 * (end - copied), "encode_ms": 1000 * (end - start)})
