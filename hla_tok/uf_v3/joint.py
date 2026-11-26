"""Differentiable RGB-to-Base recurrence for the existing UF-LD latent and D1."""

from bisect import bisect_right
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from cp_adaptok.backbones.intra_adapter import restore_rgb
from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.entropy import MULTIPLIERS, PRECISION, RADIUS
from hla_tok.uf.latent import LATENT_UNIT, UnifiedRefinement
from hla_tok.uf.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD
from hla_tok.uf_v2.decoder import SharedRefinement, reconstruction
from hla_tok.uf_v2.records import BatchRecords, quantized_symbols
from hla_tok.uf_v2.runtime import BASE_STANDARD_QP
from third_party.DCVC.src.layers.layers import QuantFunc
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.video_model_ld import DMC
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

CODED_FPS = 30
NATIVE_PIXELS = 1080 * 1920
TRAIN_BUDGETS = (2, 5, 10, 20, 40)
SEGMENTS = Segmentation(8, 8)


def qp_row(value: Tensor, qp: int, dtype: torch.dtype) -> Tensor:
    return value[qp:qp + 1, :, None, None].to(dtype)


@dataclass(frozen=True)
class BaseState:
    feature: Tensor
    memory: Tensor | None = None


@dataclass(frozen=True)
class BaseLayer:
    state: BaseState
    feature: Tensor
    context: Tensor | None
    latent: Tensor
    normalized_y: Tensor
    scale: Tensor
    yuv: Tensor
    bits: Tensor


class DifferentiableUF(nn.Module):
    def __init__(self, intra: DMCI, inter: DMC, quantizer: UnifiedRefinement, refiner: SharedRefinement) -> None:
        super().__init__()
        self.intra, self.inter, self.quantizer, self.refiner = intra, inter, quantizer, refiner

    def forward(self, rgb: Tensor, state: BaseState | None) -> BaseLayer:
        height, width = rgb.shape[-2:]
        pixels = F.pad(rgb2ycbcr(rgb) - 0.5, (0, -width % 64, 0, -height % 64), mode="replicate")
        pixels = pixels.contiguous(memory_format=torch.channels_last)
        return self.encode_intra(pixels) if state is None else self.encode_inter(pixels, state)

    def encode_intra(self, pixels: Tensor) -> BaseLayer:
        model, qp = self.intra, BASE_STANDARD_QP[0]
        y = model.enc(pixels, qp_row(model.q_scale_enc, qp, torch.bfloat16))
        z = model.hyper_enc(y)
        z_hat = QuantFunc.apply(z.float()).clamp(-127, 127)
        prior = model.y_prior_fusion(model.hyper_dec(z_hat))
        q_enc = qp_row(model.q_scale_y_enc, qp, torch.float32)
        q_dec = qp_row(model.q_scale_y_dec, qp, torch.float32)
        residual, _, latent, sigma = model.forward_prior_4x(
            y.float(), q_enc, q_dec, prior.float(), model.y_spatial_prior_reduction,
            model.y_spatial_prior_adaptor_1, model.y_spatial_prior_adaptor_2,
            model.y_spatial_prior_adaptor_3, model.y_spatial_prior)
        feature = model.dec.dec_1(latent) * qp_row(model.q_scale_dec, qp, y.dtype)
        yuv = reconstruction(model, feature, True)
        bits = model.get_y_bits(residual, sigma).sum() + model.get_z_bits(z_hat, qp).sum()
        state = BaseState(F.pixel_unshuffle(yuv, 8))
        return BaseLayer(state, feature, None, latent, y.float() * q_enc, q_dec, yuv, bits)

    def encode_inter(self, pixels: Tensor, state: BaseState) -> BaseLayer:
        model, qp = self.inter, BASE_STANDARD_QP[1]
        memory = (model.feature_adaptor_i(state.feature) if state.memory is None else
                  model.feature_adaptor_m(state.memory, state.feature))
        context = model.feature_extractor(memory)
        y = model.encoder(pixels, context, qp_row(model.q_encoder, qp, context.dtype))
        z = model.hyper_encoder(y)
        z_hat = QuantFunc.apply(z.float()).clamp(-127, 127)
        prior = model.res_prior_param_decoder(z_hat, memory, qp_row(model.q_feature, qp, memory.dtype)).float()
        _, scale, _, _ = model.separate_prior_video(prior)
        residual, _, latent, sigma = model.forward_prior_2x(y.float(), prior, model.y_spatial_prior)
        feature = model.decoder(latent, context, qp_row(model.q_decoder, qp, context.dtype))
        yuv = reconstruction(model, feature, False)
        bits = model.get_y_bits(residual, sigma).sum() + model.get_z_bits(z_hat, qp).sum()
        return BaseLayer(BaseState(feature, memory), feature, context, latent, y.float() / scale, scale, yuv, bits)

    def display(self, base: BaseLayer, delta: Tensor, intra: bool, shape: tuple[int, int]) -> Tensor:
        feature = self.refiner.refine(intra, base.feature, delta, base.context)
        yuv = reconstruction(self.intra if intra else self.inter, feature, intra)
        return restore_rgb(yuv.float(), *shape)


@dataclass(frozen=True)
class EnhancementPrefix:
    levels: NDArray[np.uint8]
    symbols: NDArray[np.int16]
    records: int
    bytes_used: int


def packed_prefix(bank: BatchRecords, intra: bool, symbols: NDArray[np.int8],
                  caps: tuple[int, ...]) -> list[EnhancementPrefix]:
    """Rank once and decode new native enhancement bytes at each prefix boundary."""
    if not caps or tuple(sorted(caps)) != caps or caps[0] < 0:
        raise ValueError("expected increasing nonnegative enhancement allowances")
    shape = (symbols.shape[1], symbols.shape[2], symbols.shape[3])
    canvas = bank.canvas(shape, intra)
    values, stats = canvas.geometry.native.gather(np.ascontiguousarray(symbols))
    scores, active, indexes = bank._priorities(stats, canvas.geometry.counts, intra)
    if bank.static is not None:
        indexes = bank.static_indexes(canvas.geometry)
    raw, ends = bank.banks[intra].pack(canvas.geometry.native, values, scores, active, indexes, True, caps[-1], bank.raw)
    boundaries = (0, *ends)
    begin = 0
    prefixes = []
    for cap in caps:
        records = bisect_right(boundaries, cap) - 1
        end = boundaries[records]
        consumed, _ = canvas.decode(bytearray(raw[begin:end]))
        if consumed != end - begin or canvas.records != records:
            raise ValueError("training prefix receiver rejected a physical enhancement boundary")
        prefixes.append(EnhancementPrefix(canvas.levels.copy(), canvas.symbols.copy(), records, end))
        begin = end
    return prefixes


def refinement(value: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
    scaled = value / step
    coarse = QuantFunc.apply(scaled)
    fine = QuantFunc.apply((scaled - coarse.detach()) * 2)
    return coarse * step, fine * (step * 0.5)


def prefix_masks(prefix: EnhancementPrefix, residual: Tensor) -> tuple[Tensor, Tensor]:
    channels, height, width = residual.shape[1:]
    bands, rows, cols = (ceil(n / 8) for n in (channels, height, width))
    levels = torch.as_tensor(prefix.levels.reshape(rows, cols, bands).transpose(2, 0, 1), device=residual.device)
    for dim in range(3):
        levels = levels.repeat_interleave(8, dim=dim)
    levels = levels[None, :channels, :height, :width]
    return levels >= 1, levels >= 2


def conditional_cost(signed: Tensor, scale: Tensor) -> Tensor:
    magnitude = signed.square().add(1e-4).sqrt()
    indexes = ((magnitude.detach().clamp_min(0.125) / scale.detach()).log2().round() + 1).clamp(0, 3)
    scale = scale * signed.new_tensor(MULTIPLIERS)[indexes.long()]
    return torch.log2(2 * scale).clamp_min(0.08) + magnitude / (scale * np.log(2))


def static_cost(signed: Tensor, cdf, level: int) -> Tensor:
    # A linear extension of the preinstalled integer CDF cost has nonzero
    # derivative between adjacent symbol locations. Escape cost is capped at
    # the last table bin and the signed bypass magnitude is charged separately.
    bands = ceil(signed.shape[1] / 8)
    frequencies = np.diff(np.array(cdf[level * bands:(level + 1) * bands]), axis=1)
    alphabet = np.arange(-RADIUS - 1, RADIUS + 2)
    indexes = np.minimum(np.abs(alphabet) * 2 - (alphabet > 0), frequencies.shape[1] - 1)
    cost = -np.log2(frequencies[:, indexes] / PRECISION)
    values = (signed + RADIUS + 1).clamp(0, len(alphabet) - 1.0001)
    left = values.detach().floor().long()
    weights = values - left
    table = signed.new_tensor(cost).repeat_interleave(8, dim=0)[:signed.shape[1]]
    rows = torch.arange(signed.shape[1], device=signed.device)[None, :, None, None]
    return table[rows, left] * (1 - weights) + table[rows, left + 1] * weights + (
        signed.abs() - RADIUS).clamp_min(0)


def coded_rate(coarse: Tensor, fine: Tensor, step: Tensor, scales: Tensor, masks: tuple[Tensor, Tensor],
               records: int, cdf=None) -> Tensor:
    terms = []
    for level, (plane, mask) in enumerate(zip((coarse, fine), masks, strict=True)):
        signed = plane / (step * (0.5 ** level))
        cost = conditional_cost(signed, scales[level]) if cdf is None else static_cost(signed, cdf, level)
        terms.append((cost * mask).sum())
    return terms[0] + terms[1] + records * (RECORD_OVERHEAD + 4) * 8

@dataclass(frozen=True)
class FrameRD:
    state: BaseState
    rgb: list[Tensor]
    budgets: list[int]
    rates: list[Tensor]
    estimated_base_bytes: int
    physical_enhancement_bytes: list[int]


def progressive_frame(model: DifferentiableUF, bank: BatchRecords, rgb: Tensor, state: BaseState | None,
                      *, budgets: tuple[int, ...] = TRAIN_BUDGETS) -> FrameRD:
    intra = state is None
    height, width = rgb.shape[-2:]
    pixels = height * width
    base = model(rgb, state)
    bits = base.bits.float() + (APP_BASE_OVERHEAD + 4) * 8
    base_bytes = ceil(float(bits.detach()) / 8)
    allowance = {budget: budget * 1_000_000 * pixels // (8 * CODED_FPS * NATIVE_PIXELS) - base_bytes
                 for budget in budgets}
    reachable = [budget for budget, cap in allowance.items() if cap >= 0]
    images = [restore_rgb(base.yuv.float(), height, width)]
    rates, actual = [bits / pixels], [0]
    if not reachable:
        return FrameRD(base.state, images, [0], rates, base_bytes, actual)

    # Hold decoded Base fixed for the enhancement STE. The Base reconstruction
    # itself has its own full gradient through the native prior and decoder.
    error = (base.normalized_y - base.latent.detach() / base.scale) / LATENT_UNIT
    step = model.quantizer.quantizer(intra).step()
    symbols = quantized_symbols(error.detach(), step.detach())
    prefixes = packed_prefix(bank, intra, symbols, tuple(allowance[b] for b in reachable))
    coarse, fine = refinement(error, step)
    unit = base.scale * LATENT_UNIT
    scales = model.quantizer.scales()
    cdf = bank.static[intra].cdf if bank.static else None
    for prefix in prefixes:
        masks = prefix_masks(prefix, error)
        rate = coded_rate(coarse, fine, step, scales, masks, prefix.records, cdf)
        if prefix.records:
            proxy = (coarse * masks[0] + fine * masks[1]) * unit
            exact = torch.as_tensor(prefix.symbols, device=error.device, dtype=torch.float32)[None]
            delta = exact * (step * unit * 0.5)
            # The forward path uses the physical decoded canvas and the
            # derivative follows the continuous residual/quantizer.
            delta = delta.detach() + (proxy - proxy.detach())
            image = model.display(base, delta, intra, (height, width))
        else:
            image = images[0]
        images.append(image)
        rates.append((bits + rate) / pixels)
        actual.append(prefix.bytes_used)
    return FrameRD(base.state, images, [0, *reachable], rates, base_bytes, actual)
