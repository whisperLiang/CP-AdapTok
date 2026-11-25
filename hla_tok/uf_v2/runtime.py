"""Single native analysis, physical prefix families, and independent D0/D1/Anchor reception."""

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import torch
from torch import Tensor

from cp_adaptok.backbones.intra_adapter import restore_rgb
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.uf.anchor import MAX_APPLICATION_BYTES, STRIDE, RefinementCodec, Segmentation, stream_tag
from hla_tok.uf.base import BaseFrame
from hla_tok.uf.latent import LATENT_UNIT, UnifiedCodec, UnifiedRefinement, normalized_error, render_latent
from hla_tok.uf.official import NativeDecoder, NativeEncoder, OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD, BaseBudgetExceeded, PrefixFamily, WireError
from hla_tok.uf_v2.decoder import SharedDecoder, SharedRefinement
from hla_tok.uf_v2.records import BatchReceiver, BatchRecords, Display, quantized_symbols

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


class AnchorFast:
    def __init__(self, assets: OfficialAssets, codec: RefinementCodec, qp: tuple[int, int]):
        self.assets, self.native, self.codec, self.qp = assets, Analysis(assets), codec, qp
        self.bank = BatchRecords(codec.records_codec.scales, codec.records_codec.segmentation)

    def reset(self) -> None:
        self.native.reset()

    def new_decoder(self) -> NativeDecoder:
        return NativeDecoder(self.assets)

    def receiver(self, decoder: NativeDecoder) -> BatchReceiver:
        def prepare(frame: BaseFrame) -> Display:
            check_base(frame, decoder, BASE_STANDARD_CAP)
            base = decoder.decode(frame)
            shape = (self.codec.model.channels, base.shape[-2] // STRIDE, base.shape[-1] // STRIDE)

            def display(delta: Tensor) -> Tensor:
                with self.codec.compute(base):
                    return self.codec.model.display(base, delta)

            return Display(base, self.bank.canvas(shape, frame.info.intra), self.codec.step, display)

        return BatchReceiver(self.codec.tag, prepare)

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: BatchReceiver) -> bool:
        return receiver.base is not None and torch.equal(encoded.reference, receiver.base)

    @torch.no_grad()
    def encode(self, rgb: Tensor, index: int, intra: bool, reset: bool = False,
               cap: int = MAX_APPLICATION_BYTES) -> Encoded:
        start = perf_counter()
        base = self.native.encode_base(rgb, index, self.qp, intra, reset, min(cap, BASE_STANDARD_CAP))
        compressed = perf_counter()
        preview = self.native.preview(base)
        with self.codec.compute(rgb):
            latent = self.codec.model.analyze(rgb, preview)
        symbols = quantized_symbols(latent, self.codec.step)
        copied = perf_counter()
        family = self.bank.pack_symbols(base, self.codec.tag, symbols, cap)
        end = perf_counter()
        return Encoded(family, preview, {"native_encode_ms": 1000 * (compressed - start),
                       "analysis_copy_ms": 1000 * (copied - compressed),
                       "group_rank_pack_ms": 1000 * (end - copied), "encode_ms": 1000 * (end - start)})


class Unified:
    def __init__(self, assets: OfficialAssets, model: UnifiedRefinement, qp: tuple[int, int],
                 refiner: SharedRefinement | None = None, bank: BatchRecords | None = None,
                 segmentation: Segmentation = Segmentation(8, 8), base_cap: int = BASE_STANDARD_CAP, family_id: str = "standard"):
        self.assets, self.native, self.qp, self.base_cap = assets, Analysis(assets), qp, base_cap
        self.model, self.refiner = model, refiner
        self.steps = {intra: model.quantizer(intra).step().detach() for intra in (True, False)}
        legacy = UnifiedCodec(model, segmentation)
        self.bank = bank or BatchRecords(legacy.bank.scales, segmentation)
        canonical = (qp, base_cap, family_id) == (BASE_STANDARD_QP, BASE_STANDARD_CAP, "standard")
        if refiner is None and bank is None and canonical:
            self.tag = legacy.tag
        else:
            weights = self.bank.sensitivity
            digest = sha256(b"".join(weights[i].astype("<f8").tobytes() for i in (True, False))).hexdigest() if (
                weights is not None) else None
            self.tag = stream_tag(refiner if refiner is not None else model,
                                  f"uf-v2/{family_id}/{qp}/{base_cap}/{legacy.tag.hex()}/{self.bank.segmentation}/"
                                  f"{self.bank.static}/{self.bank.raw}/{digest}")

    def reset(self) -> None:
        self.native.reset()

    def new_decoder(self) -> NativeDecoder:
        return SharedDecoder(self.assets) if self.refiner is not None else NativeDecoder(self.assets)

    def receiver(self, decoder: NativeDecoder) -> BatchReceiver:
        return BatchReceiver(self.tag, lambda frame: self.prepare(decoder, frame))

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: BatchReceiver) -> bool:
        return torch.equal(encoded.reference, decoder.decoded_latent())

    def _display(self, decoder: NativeDecoder, frame: BaseFrame) -> Display:
        if self.refiner is not None and isinstance(decoder, SharedDecoder):
            snapshot = decoder.shared_base(frame)
            model = self.refiner
            return self._make_display(frame, snapshot.rgb, snapshot.scale,
                                      lambda delta: decoder.display(snapshot, model, delta))
        if self.refiner is not None:
            raise TypeError("D1 requires its own independent shared-feature decoder")
        base = decoder.decode(frame)
        assert decoder.last_net is not None
        latent, scale, *features = decoder.last_net.proxy.uf_display_snapshot()
        context = None if frame.info.intra else features[0]

        def display(delta: Tensor) -> Tensor:
            y = (latent.float() + delta).to(latent.dtype).contiguous(memory_format=torch.channels_last)
            return render_latent(decoder, frame.info, y, context)

        return self._make_display(frame, base, scale, display)

    def _make_display(self, frame, base, scale, display) -> Display:
        info = frame.info
        steps = self.steps[info.intra]
        padded = tuple((size + post) // 16 for size, post in zip(info.dimensions, info.native_padding, strict=True))
        canvas = self.bank.canvas((steps.shape[1], *padded), info.intra)
        return Display(base, canvas, scale.float() * steps * LATENT_UNIT, display)

    @torch.no_grad()
    def prepare(self, decoder: NativeDecoder, frame: BaseFrame) -> Display:
        check_base(frame, decoder, self.base_cap)
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
