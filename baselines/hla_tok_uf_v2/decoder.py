"""Refine the official I/P reconstruction features while preserving native references."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from baselines.dcvc_uf.official import NativeDecoder, OfficialAssets
from common.video import restore_rgb
from third_party.DCVC.src.layers.layers import DepthConvBlock
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.image_model import g_ch_enc_dec as INTRA_FEATURES
from third_party.DCVC.src.models.image_model import g_ch_y as INTRA_LATENT
from third_party.DCVC.src.models.video_model_ld import DMC
from third_party.DCVC.src.models.video_model_ld import g_ch_d as INTER_FEATURES
from third_party.DCVC.src.models.video_model_ld import g_ch_m as CONTEXT_CHANNELS
from third_party.DCVC.src.models.video_model_ld import g_ch_y as INTER_LATENT


class FeatureRefinement(nn.Module):
    def __init__(self, latent: int, features: int, context: int = 0, width: int = 32):
        super().__init__()
        self.up = nn.Conv2d(latent, 4 * width, 1)
        self.base = nn.Conv2d(features, width, 1, bias=False)
        self.context = nn.Conv2d(context, width, 1, bias=False) if context else None
        self.local = nn.Sequential(DepthConvBlock(width, width, dcb2=True),
                                   DepthConvBlock(width, width, dcb2=True))
        self.out = nn.Conv2d(width, features, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, base: Tensor, delta: Tensor, context: Tensor | None = None) -> Tensor:
        projected = F.pixel_shuffle(self.up(delta), 2) + self.base(base)
        if self.context is not None:
            if context is None:
                raise ValueError("P refinement requires the current frame context")
            projected = projected + self.context(context)
        return base + self.out(self.local(projected)).to(base.dtype)


class SharedRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra = FeatureRefinement(INTRA_LATENT, INTRA_FEATURES)
        self.inter = FeatureRefinement(INTER_LATENT, INTER_FEATURES, CONTEXT_CHANNELS)

    def branch(self, intra: bool) -> FeatureRefinement:
        return self.intra if intra else self.inter

    def refine(self, intra: bool, base: Tensor, delta: Tensor, context: Tensor | None) -> Tensor:
        return self.branch(intra)(base, delta, context)


@dataclass(frozen=True)
class SharedBase:
    info: FrameInfo
    rgb: Tensor
    scale: Tensor
    feature: Tensor
    context: Tensor | None


def reconstruction(network: DMCI | DMC, feature: Tensor, intra: bool) -> Tensor:
    if intra:
        return F.pixel_shuffle(network.dec.dec_2(feature), 8)
    return network.recon_head(feature)


class SharedDecoder(NativeDecoder):
    def __init__(self, assets: OfficialAssets):
        super().__init__(assets)
        self.intra.proxy.ufv2_capture_display()

    @torch.no_grad()
    def shared_base(self, frame: BaseFrame) -> SharedBase:
        rgb = self.decode(frame)
        assert self.last_net is not None
        with torch.cuda.stream(self.stream):
            scale, feature, *context = self.last_net.proxy.ufv2_display_snapshot()
        torch.cuda.current_stream().wait_stream(self.stream)
        return SharedBase(frame.info, rgb, scale, feature, context[0] if context else None)

    def display(self, base: SharedBase, model: SharedRefinement, delta: Tensor) -> Tensor:
        intra = base.info.intra
        features = model.refine(intra, base.feature, delta.to(base.feature.dtype), base.context)
        pixels = reconstruction(self.intra if intra else self._video(), features, intra)
        return restore_rgb(pixels.float(), *base.info.dimensions)
