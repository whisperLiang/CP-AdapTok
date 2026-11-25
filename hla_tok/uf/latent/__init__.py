"""Refine the official normalized latent without feeding display pixels back."""

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from cp_adaptok.backbones.intra_adapter import restore_rgb
from hla_tok.uf.anchor import (
    DisplayFrame,
    FrameReceiver,
    IntegerCanvas,
    RecordCodec,
    Segmentation,
    stream_tag,
)
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.official import NativeDecoder, NativeEndpoint
from hla_tok.uf.refinement import RefinementQuantizer
from hla_tok.uf.wire import Enhancement
from third_party.DCVC.src.models.image_model import g_ch_y as INTRA_CHANNELS
from third_party.DCVC.src.models.video_model_ld import g_ch_y as VIDEO_CHANNELS

LATENT_UNIT = 4


def normalized_error(encoded_y: Tensor, decoded_y: Tensor, native_step: Tensor) -> Tensor:
    return (encoded_y.float() - decoded_y.float() / native_step.float()) / LATENT_UNIT


class UnifiedRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intra = RefinementQuantizer(INTRA_CHANNELS, initial_step=0.125)
        self.inter = RefinementQuantizer(VIDEO_CHANNELS, initial_step=0.125)
        self.log_scale = nn.Parameter(torch.tensor([0.25, -0.75]))

    def quantizer(self, intra: bool) -> RefinementQuantizer:
        return self.intra if intra else self.inter

    def scales(self) -> Tensor:
        return self.log_scale.float().clamp(-3, 3).exp()


def observe(endpoint: NativeEndpoint, intra: bool, method: str) -> list[Tensor]:
    network = endpoint.intra if intra else endpoint.video
    if network is None:
        raise ValueError("observation requires a video endpoint")
    with torch.cuda.stream(endpoint.stream):
        output = getattr(network.proxy, method)()
    endpoint.stream.synchronize()
    return output


def render_latent(decoder: NativeDecoder, info: FrameInfo, y_base: Tensor, context: Tensor | None) -> Tensor:
    if info.intra:
        quant = decoder.intra.q_scale_dec[info.qp:info.qp + 1, :, None, None]
        pixels = decoder.intra.dec(y_base, quant)
    else:
        video = decoder.video
        if video is None or context is None:
            raise ValueError("inter display requires native context")
        quant = video.q_decoder[info.qp:info.qp + 1, :, None, None]
        pixels, _ = video.get_recon_and_feature(y_base, context, quant)
    return restore_rgb(pixels.float(), *info.dimensions)


class UnifiedCodec:
    def __init__(self, model: UnifiedRefinement, segmentation: Segmentation = Segmentation(8, 8)):
        self.model = model
        self.bank = RecordCodec(tuple(model.scales().detach().cpu().tolist()), segmentation)
        self.tag = stream_tag(model, f"uf-latent-v1/{segmentation}/{self.bank.tables.cdf}")

    def records(self, intra: bool, encoded_y: Tensor, y_base: Tensor, native_step: Tensor) -> Iterator[Enhancement]:
        residual = normalized_error(encoded_y, y_base, native_step)
        return self.bank.records(*self.model.quantizer(intra).symbols(residual))

    def receiver(self, native: NativeDecoder) -> FrameReceiver:
        return FrameReceiver(self.tag, lambda frame: self.prepare(native, frame))

    @torch.no_grad()
    def prepare(self, native: NativeDecoder, frame: BaseFrame) -> DisplayFrame:
        base = native.decode(frame)
        intra = frame.info.intra
        y_base, step, *features = observe(native, intra, "uf_display_snapshot")
        context = None if intra else features[0]
        step = step.float() * self.model.quantizer(intra).step() * LATENT_UNIT
        channels, height, width = y_base.shape[-3:]
        canvas = IntegerCanvas(self.bank, (channels, height, width))

        @torch.no_grad()
        def display(delta: Tensor) -> Tensor:
            latent = (y_base.float() + delta).to(y_base.dtype).contiguous(memory_format=torch.channels_last)
            return render_latent(native, frame.info, latent, context)

        return DisplayFrame(base, canvas, step, display)
