"""Continue D1 with a differentiable rate proxy and exact Base replay at each node."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cp_adaptok.backbones.intra_adapter import restore_rgb
from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import LATENT_UNIT, UnifiedRefinement
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD
from hla_tok.uf_v2.decoder import SharedDecoder, SharedRefinement, reconstruction
from hla_tok.uf_v2.native_weights import NativeAdjustments
from hla_tok.uf_v2.records import BatchReceiver
from hla_tok.uf_v2.runtime import BASE_STANDARD_QP, Unified
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.train import Crop, crop_center, layout
from scripts.hla_tok.v2.protocol import ROOT, setup
from third_party.DCVC.src.layers.layers import bit_estimator_z_fwd

CONFIG = ROOT / "configs/hla_tok/uf_v2/joint.json"


@dataclass(frozen=True)
class Example:
    intra: bool
    rgb: Tensor
    feature: Tensor
    ctx: Tensor | None
    native_y: Tensor
    original: Tensor
    scale: Tensor
    y_symbols: Tensor | None
    sigma: Tensor | None
    selection: Tensor
    byte_counts: list[int]
    record_counts: list[int]


@torch.no_grad()
def prepare_frame(codec: Unified, decoder: SharedDecoder, rgb: Tensor, index: int) -> Example:
    intra, reset = schedule(index)
    encoded = codec.encode(rgb, index, intra, reset)
    family = encoded.family
    old = codec.native.proxy(intra).uf_encode_snapshot()
    original, y, scale = old[:3]
    if intra:
        y_symbols, sigma = None, None
    else:
        y_symbols, prior = old[4:6]
        sigma = layout(prior[:, y.shape[1]:2 * y.shape[1]].clone())

    cache = []

    def capture(frame):
        base = decoder.shared_base(frame)
        cache.append(base)
        return codec._make_display(frame, base.rgb, base.scale, lambda delta: delta)

    receive = BatchReceiver(codec.tag, capture)
    receive.feed(family.raw[:family.base_end])
    if receive.decode_available() != 1 or not torch.equal(y, decoder.decoded_latent()):
        raise ValueError("independent training Base latent mismatch")
    bytes_per_prefix, counts, masks = [family.base_end], [0], []
    assert receive.canvas is not None
    begin = family.base_end
    shape = receive.canvas.geometry.groups.grid
    for cap in BUDGETS.values():
        raw, count = family.prefix(cap)
        receive.feed(raw[begin:])
        receive.decode_available()
        receive.finish()
        masks.append(receive.canvas.levels.reshape(shape[1], shape[2], shape[0]).transpose(2, 0, 1).copy())
        bytes_per_prefix.append(len(raw))
        counts.append(count)
        begin = len(raw)
    selection = torch.from_numpy(np.stack(masks)).to(rgb.device)
    state = cache[0]
    return Example(intra, rgb.mul(255).round().byte(), state.feature, state.context, y, original, scale,
                   y_symbols, sigma, selection, bytes_per_prefix, counts)


@torch.no_grad()
def cache_closed_loop(assets, quantizer, refiner, adjustments, reader, clips):
    codec = Unified(assets, quantizer, BASE_STANDARD_QP, refiner, adjustments=adjustments)
    decoder = codec.new_decoder()
    assert isinstance(decoder, SharedDecoder)
    decoder.intra.requires_grad_(False)
    decoder._video().requires_grad_(False)
    clip_cache = []
    for clip in clips:
        if clip.split != "train":
            raise PermissionError("joint fit requires Train-only inputs")
        codec.reset()
        decoder.reset()
        cached = [prepare_frame(codec, decoder, source_frame(reader, clip, i), i) for i in range(clip.length)]
        clip_cache.append(cached)
        print("joint closed-loop cache", clip.video, len(cached), flush=True)
    return decoder, clip_cache


def ste_round(value: Tensor) -> Tensor:
    return value + (value.round() - value).detach()


def stack(frames: list[Example], field: str, y: slice, x: slice) -> Tensor:
    return layout(torch.cat([getattr(frame, field)[..., y, x] for frame in frames]))


def selections(frames, crop, dtype):
    # Mask coordinates match the eight-channel / eight-cell physical segment layout.
    coarse = torch.stack([frame.selection for frame in frames], dim=1)
    tiled = coarse.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)
    return tiled[..., crop.y, crop.x].to(dtype)


@dataclass(frozen=True)
class NativeProxy:
    decoder: SharedDecoder
    live: NativeAdjustments
    encoded: NativeAdjustments

    def predict(self, original, y, scale, feature, ctx, frames, crop):
        candidate = original * (self.live.encoder_gain() / self.encoded.encoder_gain())[None, :, None, None]
        pivots = y / scale
        change = ste_round(candidate - pivots) - (original - pivots).round()
        native_base = y + change * scale
        q = BASE_STANDARD_QP[1]
        video = self.decoder._video()
        gain = video.q_decoder[q:q + 1, :, None, None]
        with torch.no_grad():
            frozen = video.decoder(layout(y.half()), ctx, gain)
        corrected = video.decoder(layout(native_base.half()), ctx, gain)
        feature = feature + (corrected - frozen)
        rate = self.rate(video, q, candidate, pivots, frames, crop)
        return candidate, native_base, feature, rate

    def rate(self, video, qp, candidate, pivots, frames, crop):
        symbols = stack(frames, "y_symbols", crop.y, crop.x).float()
        sigma = stack(frames, "sigma", crop.y, crop.x).float().clamp_min(0.01)
        y_bits = video.get_y_bits(candidate - pivots + symbols, sigma).sum((1, 2, 3))
        original_units = candidate * stack(frames, "scale", crop.y, crop.x).float()
        pad_h, pad_w = (-original_units.shape[-2]) % 4, (-original_units.shape[-1]) % 4
        z = video.hyper_encoder(layout(F.pad(original_units.half(), (0, pad_w, 0, pad_h), mode="replicate")))
        h, b, a = video.bit_estimator_z.get_hba(qp, dtype=torch.float32)
        bias = self.live.hyper_bias() - self.encoded.hyper_bias()
        b = b + F.pad(bias[None, :, None], (3, 0))
        z_bits = video.probs_to_bits(bit_estimator_z_fwd(ste_round(z.float()), h, b, a)).sum((1, 2, 3))
        pixels = candidate.shape[-2] * candidate.shape[-1] * 256
        return (y_bits + z_bits) / pixels + APP_BASE_OVERHEAD * 8 / (1920 * 1080)

    def display(self, features, intra):
        if intra:
            return reconstruction(self.decoder.intra, features, True)
        head = self.decoder._video().recon_head
        output = head.head(head.conv(features))
        gain = self.live.reconstruction_gain() / self.encoded.reconstruction_gain()
        return F.pixel_shuffle((output.float() * gain[None, :, None, None]).half(), 8)


def enhancement_proxy(quantizer, intra, original, y, scale, selection, frames):
    error = (original - y / scale) / LATENT_UNIT
    coarse, fine = quantizer.quantizer(intra)(error)
    mask0, mask1 = selection > 0, selection > 1
    effective = coarse[None] * mask0 + fine[None] * mask1
    units = (scale * LATENT_UNIT)[None] * effective
    q = quantizer.quantizer(intra).step()[None]
    p0, p1 = quantizer.scales()
    cost0 = torch.log2(2 * p0).clamp_min(0.08) + coarse[None].abs() / (q * p0 * np.log(2))
    cost1 = torch.log2(2 * p1).clamp_min(0.08) + fine[None].abs() / (q * 0.5 * p1 * np.log(2))
    pixels = original.shape[-2] * original.shape[-1] * 256
    bits = (cost0 * mask0 + cost1 * mask1).sum((2, 3, 4)) / pixels
    overhead = torch.tensor([e.record_counts[1:] for e in frames], dtype=torch.float32, device=y.device).T
    return units, bits + overhead * (RECORD_OVERHEAD + 4) * 8 / (1920 * 1080)


def losses(refiner, quantizer, native, lpips, frames, crop, weights):
    intra, batch = frames[0].intra, len(frames)
    feature = stack(frames, "feature", crop.fy, crop.fx)
    source = stack(frames, "rgb", crop.ry, crop.rx).float() / 255
    y = stack(frames, "native_y", crop.y, crop.x).float()
    original = stack(frames, "original", crop.y, crop.x).float()
    if intra:
        scale = torch.cat([e.scale for e in frames]).float()
        native_bits = torch.tensor([e.byte_counts[0] for e in frames], device=y.device) * (8 / (1920 * 1080))
        ctx = None
    else:
        scale = stack(frames, "scale", crop.y, crop.x).float()
        ctx = stack(frames, "ctx", crop.fy, crop.fx)
        original, y, feature, native_bits = native.predict(original, y, scale, feature, ctx, frames, crop)
    masks = selections(frames, crop, torch.float32)
    deltas, bits = enhancement_proxy(quantizer, intra, original, y, scale, masks, frames)
    k = deltas.shape[0]
    ctx = layout(ctx.repeat(k, 1, 1, 1)) if ctx is not None else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enhanced = refiner.refine(intra, layout(feature.repeat(k, 1, 1, 1)),
                                  layout(deltas.flatten(0, 1).half()), ctx)
    reconstructed = native.display(layout(torch.cat((feature, enhanced))), intra)
    height, width = source.shape[-2:]
    pred = restore_rgb(crop_center(reconstructed, height, width).float(), height, width)
    target = source.repeat(k + 1, 1, 1, 1)
    error = pred - target
    mse = error.square().mean((1, 2, 3))
    perceptual = lpips(pred.mul(2).sub(1), target.mul(2).sub(1)).flatten()
    luma = 0.299 * error[:, :1] + 0.587 * error[:, 1:2] + 0.114 * error[:, 2:3]
    high = (luma - F.avg_pool2d(luma, 9, stride=1, padding=4, count_include_pad=False)).abs().mean((1, 2, 3))
    distortion = (weights["rgb_mse"] * mse + weights["lpips"] * perceptual
                  + weights["high_frequency"] * high).reshape(k + 1, batch)
    rate = torch.cat((native_bits[None], native_bits[None] + bits))
    rd = distortion + weights["rate"] * rate
    monotonic = F.relu(distortion[1:] - distortion[:-1]).mean()
    loss = rd[1:].mean() + weights["base"] * rd[0].mean() + weights["monotonic"] * monotonic
    measured = torch.tensor([e.byte_counts for e in frames], device=y.device).T * (8 / (1920 * 1080))
    reported = torch.stack((loss, mse.mean(), perceptual.mean(), rate.mean(), measured.mean(), monotonic)).detach()
    return loss, reported


def effective_gradient(parameters):
    return sum(p.grad.detach().float().square().sum().item() for p in parameters if p.grad is not None) ** 0.5


def step(modules, native, lpips, optimizer, frames, crop, config):
    refiner, quantizer, adjustments = modules
    optimizer.zero_grad(set_to_none=True)
    stats = []
    for intra in (False, True):
        selected = [f for f in frames if f.intra == intra]
        if not selected:
            continue
        weight = len(selected) / len(frames)
        loss, metrics = losses(refiner, quantizer, native, lpips, selected, crop, config["weights"])
        (loss * weight).backward()
        stats.append(metrics * weight)
    native_grad = effective_gradient(adjustments.parameters())
    quant_grad = effective_gradient(quantizer.parameters())
    grad = nn.utils.clip_grad_norm_(
        [p for model in modules for p in model.parameters()], config["clip_grad"], error_if_nonfinite=True)
    if float(grad) == 0:
        raise RuntimeError("joint optimizer did not obtain a valid gradient")
    optimizer.step()
    values = torch.stack(stats).sum(0).tolist()
    return dict(zip(("loss", "mse", "lpips", "estimated_bpp", "actual_cache_bpp", "monotonic"), values, strict=True)) | {
        "gradient_norm": grad.item(), "native_gradient_norm": native_grad, "quantizer_gradient_norm": quant_grad}


def model_and_state(args, config, signature):
    refiner = SharedRefinement().cuda().to(memory_format=torch.channels_last)
    quantizer = UnifiedRefinement().cuda()
    adjustments = NativeAdjustments().cuda()
    state = torch.load(args.resume, map_location="cpu", weights_only=True)
    refiner.load_state_dict(state["refiner"])
    quantizer.load_state_dict(state["quantizer"])
    groups = [{"params": model.parameters(), "lr": config["learning_rates"][name]}
              for name, model in zip(("refiner", "quantizer", "native"), (refiner, quantizer, adjustments), strict=True)]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    if "native_adjustments" in state:
        if state["signature"] != signature:
            raise ValueError("joint continuation changed its source, data, configuration or seed")
        adjustments.load_state_dict(state["native_adjustments"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    elif int(state["step"]) != 2000:
        raise ValueError("joint training starts from a verified 2,000-step P1 checkpoint")
    return (refiner, quantizer, adjustments), optimizer, int(state["step"])


def activate(adjustments, index):
    adjustments.encoder_logits.requires_grad_(index >= 2100)
    adjustments.reconstruction_logits.requires_grad_(index >= 2600)
    adjustments.hyper_logits.requires_grad_(index >= 2600)


def write_checkpoint(args, modules, optimizer, signature, number):
    refiner, quantizer, adjustments = modules
    path = args.checkpoints / f"d1-joint-seed{args.seed}-step{number}.pt"
    temp = path.with_suffix(".tmp")
    torch.save({"refiner": refiner.state_dict(), "quantizer": quantizer.state_dict(),
                "native_adjustments": adjustments.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "signature": signature,
                "step": number, "next_data_position": number}, temp)
    temp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2001, 4000, 6000, 8000), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup(args.seed)
    config = json.loads(CONFIG.read_text())
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    signature = {"config": sha256_file(CONFIG), "source": sha256_file(Path(__file__)), "seed": args.seed,
                 "manifests": reader.hashes, "weights": [sha256_file(p) for p in (assets.intra, assets.ld)]}
    modules, optimizer, first = model_and_state(args, config, signature)
    if args.steps <= first or args.steps - first > 2000:
        raise ValueError("run one 2,000-step node at a time")
    refiner, quantizer, adjustments = modules
    encoded = NativeAdjustments().cuda().requires_grad_(False)
    encoded.load_state_dict(adjustments.state_dict())
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    lpips = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    decoder, cache = cache_closed_loop(assets, quantizer, refiner, encoded, reader, plans["fit"])
    native = NativeProxy(decoder, adjustments, encoded)
    cdf_digest = hashlib.sha256(b"".join(array.tobytes()
        for array in decoder._video().bit_estimator_z.get_cdf_info())).hexdigest()
    cache_seconds = perf_counter() - start
    trace = []
    unroll = config["unroll"]
    for index in range(first, args.steps):
        activate(adjustments, index)
        rng = random.Random((args.seed << 24) + index)
        clip = rng.choice(cache)
        offset = 0 if index % 4 == 0 else rng.randrange(0, len(clip) - unroll + 1)
        frames = clip[offset:offset + unroll]
        crop = Crop.sample(rng, config["crop_hw"], config["guard_latent_cells"])
        tick = perf_counter()
        metrics = step(modules, native, lpips, optimizer, frames, crop, config)
        number = index + 1
        trace.append(metrics | {"step": number, "step_s": perf_counter() - tick, "unroll": unroll})
        if number % 100 == 0 or number == args.steps:
            print(number, {key: round(value, 5) for key, value in metrics.items()}, flush=True)
    path = write_checkpoint(args, modules, optimizer, signature, args.steps)
    write_csv(args.output / f"steps-{first + 1}-{args.steps}.csv", trace)
    write_json(args.output / f"step-{args.steps}.json", {
        "checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "optimizer_steps": args.steps,
        "added_steps": args.steps - first, "seed": args.seed, "unroll": unroll, "rate_proxy": config["rate_proxy"],
        "cache_native_z_cdf_sha256": cdf_digest, "read_counts": reader.counts, "cache_seconds": cache_seconds,
        "gpu_hours": (perf_counter() - start) / 3600, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "train_frame_instances": unroll * (args.steps - first), "resume": str(args.resume),
        "mean_step_s": sum(m["step_s"] for m in trace) / len(trace), "signature": signature,
        "native_adjustment_sha256": adjustments.digest(), "qps": BASE_STANDARD_QP,
    })


if __name__ == "__main__":
    main()
