"""Correct DMCI continuous scales and measure the lossy 256->128 boundary.

No native entropy prior is used by this continuous Gate 0 control. Both official
backbones remain frozen; only the explicitly separate affine adapters can train.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aip_vtok.decoder.causal_state import FrameContext
from cp_adaptok.backbones.intra_adapter import IntraBackboneAdapter, prepare_rgb, restore_rgb
from cp_adaptok.backbones.ld_adapter import CausalLDBackboneAdapter


@dataclass(frozen=True)
class Geometry:
    height: int = 1080
    width: int = 1920

    def validate(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, self.height, self.width):
            raise ValueError(f"expected native RGB [B,3,{self.height},{self.width}]")
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all():
            raise ValueError("RGB must be finite floating point")
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB must be in [0,1]")

    def pad(self, rgb):
        self.validate(rgb)
        return F.pad(rgb, (0, (-self.width) % 64, 0, (-self.height) % 64), mode="replicate")

    def crop(self, rgb):
        return rgb[..., : self.height, : self.width]


def continuous_intra_latent(latent, q_enc, q_dec):
    """Retain both official forward_prior_4x scales while bypassing rounding."""
    return latent * q_enc * q_dec


def channel_samples(latents):
    return torch.cat([y.detach().movedim(1, -1).reshape(-1, y.shape[1]).double() for y in latents])


class ChannelAlignment(nn.Module):
    """Whitened PCA for I, reversible diagonal normalization for P.

    The inverse I projection is exact on its rank-128 subspace only. The missing
    128 dimensions are never hidden in a side tensor or a receiver-only bypass.
    """

    def __init__(self):
        super().__init__()
        self.intra_down = nn.Conv2d(256, 128, 1)
        self.intra_up = nn.Conv2d(128, 256, 1)
        self.register_buffer("p_mean", torch.zeros(1, 128, 1, 1))
        self.register_buffer("p_std", torch.ones(1, 128, 1, 1))
        with torch.no_grad():
            basis = torch.eye(256, 128)
            self.intra_down.weight.copy_(basis.T[:, :, None, None])
            self.intra_down.bias.zero_()
            self.intra_up.weight.copy_(basis[:, :, None, None])
            self.intra_up.bias.zero_()

    @torch.no_grad()
    def fit_intra(self, latents):
        samples = channel_samples(latents)
        mean = samples.mean(0)
        centered = samples - mean
        covariance = centered.T @ centered / samples.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -128:].flip(1)
        std = eigenvalues[-128:].flip(0).clamp_min(1e-8).sqrt()
        down = basis.T / std[:, None]
        self.intra_down.weight.copy_(down[:, :, None, None])
        self.intra_down.bias.copy_(-down @ mean)
        self.intra_up.weight.copy_((basis * std)[:, :, None, None])
        self.intra_up.bias.copy_(mean)
        return {
            "retained_variance": float(eigenvalues[-128:].sum() / eigenvalues.sum()),
            "samples": samples.shape[0],
            "rank_limit": 128,
        }

    @torch.no_grad()
    def fit_predictive(self, latents):
        samples = channel_samples(latents)
        self.p_mean.copy_(samples.mean(0)[None, :, None, None])
        self.p_std.copy_(samples.std(0, correction=0).clamp_min(1e-4)[None, :, None, None])

    def align(self, latent, intra):
        return self.intra_down(latent) if intra else (latent - self.p_mean) / self.p_std

    def restore(self, latent, intra):
        return self.intra_up(latent) if intra else latent * self.p_std + self.p_mean


class AlignedDCVC(nn.Module):
    def __init__(self, intra_checkpoint, ld_checkpoint, qp=32, gop=32, geometry=None):
        super().__init__()
        self.intra = IntraBackboneAdapter(intra_checkpoint).eval().requires_grad_(False)
        self.ld = CausalLDBackboneAdapter(ld_checkpoint).eval().requires_grad_(False)
        self.alignment = ChannelAlignment()
        self.qp, self.gop, self.geometry = qp, gop, geometry or Geometry()

    def train(self, mode=True):
        super().train(mode)
        self.intra.eval()
        self.ld.eval()
        return self

    def begin_frame(self, state, frame_id):
        if state.is_intra(frame_id, self.gop):
            return FrameContext(frame_id, True, None, None)
        if state.memory is None:
            memory = self.ld.model.feature_adaptor_i(state.feature)
        else:
            memory = self.ld.model.feature_adaptor_m(state.memory, state.feature)
        context = self.ld.model.feature_extractor(memory)
        return FrameContext(frame_id, False, context, memory)

    def native_latent(self, rgb, context):
        self.geometry.validate(rgb)
        x = prepare_rgb(rgb).to(memory_format=torch.channels_last)
        if not context.intra:
            return self.ld.model.encoder(x, context.context, self.ld.quant("q_encoder", self.qp))
        latent = self.intra.model.enc(x, self.intra.quant("q_scale_enc", self.qp))
        return continuous_intra_latent(
            latent, self.intra.quant("q_scale_y_enc", self.qp), self.intra.quant("q_scale_y_dec", self.qp)
        )

    def decode_native(self, latent, context, checkpointed=False):
        def decode(y):
            if context.intra:
                yuv = self.intra.model.dec(y, self.intra.quant("q_scale_dec", self.qp)).clamp(-0.5, 0.5)
                return yuv, F.pixel_unshuffle(yuv, 8)
            feature = self.ld.model.decoder(y, context.context, self.ld.quant("q_decoder", self.qp))
            return self.ld.model.recon_head(feature), feature

        yuv, feature = checkpoint(decode, latent, use_reentrant=False) if checkpointed else decode(latent)
        return restore_rgb(yuv, self.geometry.height, self.geometry.width), feature

    def aligned_latent(self, rgb, context):
        return self.alignment.align(self.native_latent(rgb, context), context.intra)

    def decode_aligned(self, latent, context, checkpointed=False):
        return self.decode_native(self.alignment.restore(latent, context.intra), context, checkpointed)
