"""Fit only D1 on independent, byte-decoded native Base histories and frozen D0 targets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn

from cp_adaptok.backbones.intra_adapter import restore_rgb
from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedRefinement
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.decoder import SharedDecoder, SharedRefinement, reconstruction
from hla_tok.uf_v2.runtime import Unified
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import ROOT, setup

CONFIG = ROOT / "configs/hla_tok/uf_v2/pilot.json"


@dataclass(frozen=True)
class Example:
    intra: bool
    rgb: Tensor
    feature: Tensor
    context: Tensor | None
    deltas: list[Tensor]
    teachers: list[Tensor]
    byte_counts: list[int]


def layout(value: Tensor) -> Tensor:
    return value.contiguous(memory_format=torch.channels_last)


def d0_features(decoder: SharedDecoder, intra: bool, qp: int, y: Tensor, ctx: Tensor | None) -> Tensor:
    if intra:
        native = decoder.intra
        return native.dec.dec_1(y) * native.q_scale_dec[qp:qp + 1, :, None, None]
    native = decoder._video()
    return native.decoder(y, ctx, native.q_decoder[qp:qp + 1, :, None, None])


@torch.no_grad()
def collect_examples(assets, quantizer, reader, clips, budgets, qps):
    refiner = SharedRefinement().cuda().half().to(memory_format=torch.channels_last)
    codec = Unified(assets, quantizer, qps, refiner)
    decoder = codec.new_decoder()
    assert isinstance(decoder, SharedDecoder)
    clips_data = []
    for clip in clips:
        if clip.split != "train":
            raise PermissionError("pilot fit requires Train-only inputs")
        codec.reset()
        decoder.reset()
        examples = []
        for index in range(clip.length):
            intra, reset = schedule(index)
            rgb = source_frame(reader, clip, index)
            output = codec.encode(rgb, index, intra, reset)
            family = output.family
            bases = []

            def capture(frame):
                base = decoder.shared_base(frame)
                bases.append(base)
                return codec._make_display(frame, base.rgb, base.scale, lambda delta: delta)

            from hla_tok.uf_v2.records import BatchReceiver
            receiver = BatchReceiver(codec.tag, capture)
            receiver.feed(family.raw[:family.base_end])
            if receiver.decode_available() != 1 or not torch.equal(decoder.decoded_latent(), output.reference):
                raise RuntimeError("fit cache base differs from independently decoded native state")
            base = bases[0]
            y = decoder.decoded_latent().float()
            deltas, targets, byte_counts = [], [], []
            start = family.base_end
            assert receiver.canvas is not None and receiver.frame is not None
            for budget in budgets:
                raw, _ = family.prefix(BUDGETS[budget])
                receiver.feed(raw[start:])
                receiver.decode_available()
                receiver.finish()
                delta = receiver.canvas.delta(receiver.frame.step)
                d0 = d0_features(decoder, intra, base.info.qp, layout((y + delta).half()), base.context)
                deltas.append(layout(delta.half()))
                targets.append(layout(d0))
                byte_counts.append(len(raw))
                start = len(raw)
            examples.append(Example(intra, rgb.mul(255).round().to(torch.uint8), base.feature,
                                    base.context, deltas, targets, byte_counts))
        clips_data.append(examples)
        print("fit cache", clip.video, len(examples), flush=True)
    return decoder, clips_data


@dataclass(frozen=True)
class Crop:
    y: slice
    x: slice
    fy: slice
    fx: slice
    ry: slice
    rx: slice

    @classmethod
    def sample(cls, rng: random.Random, dimensions: list[int], guard: int):
        h, w = (n // 16 for n in dimensions)
        y = rng.randrange(guard, 67 - h - guard + 1)
        x = rng.randrange(guard, 120 - w - guard + 1)
        ys, xs = (y - guard, y + h + guard), (x - guard, x + w + guard)
        return cls(slice(*ys), slice(*xs), slice(2 * ys[0], 2 * ys[1]), slice(2 * xs[0], 2 * xs[1]),
                   slice(16 * y, 16 * (y + h)), slice(16 * x, 16 * (x + w)))


def collate(examples: list[Example], crop: Crop):
    feature = layout(torch.cat([ex.feature[..., crop.fy, crop.fx] for ex in examples]))
    contexts = [ex.context[..., crop.fy, crop.fx] for ex in examples if ex.context is not None]
    contexts_batched = layout(torch.cat(contexts)) if contexts else None
    pixels = layout(torch.cat([ex.rgb[..., crop.ry, crop.rx] for ex in examples]).float() / 255)
    prefixes = len(examples[0].deltas)
    delta = layout(torch.cat([ex.deltas[n][..., crop.y, crop.x] for n in range(prefixes) for ex in examples]))
    teacher = layout(torch.cat([ex.teachers[n][..., crop.fy, crop.fx] for n in range(prefixes) for ex in examples]))
    context = layout(contexts_batched.repeat(prefixes, 1, 1, 1)) if contexts_batched is not None else None
    return layout(feature.repeat(prefixes, 1, 1, 1)), context, layout(pixels.repeat(prefixes, 1, 1, 1)), delta, teacher


def crop_center(tensor: Tensor, height: int, width: int) -> Tensor:
    y, x = (tensor.shape[-2] - height) // 2, (tensor.shape[-1] - width) // 2
    return tensor[..., y:y + height, x:x + width]


def loss_for_frames(refiner: SharedRefinement, decoder: SharedDecoder, lpips: nn.Module, examples: list[Example],
                    crop: Crop, config: dict) -> tuple[Tensor, Tensor]:
    intra = examples[0].intra
    feature, context, rgb, delta, teacher = collate(examples, crop)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student = refiner.refine(intra, feature, delta, context)
    head = decoder.intra if intra else decoder._video()
    rh, rw = rgb.shape[-2:]
    rec = reconstruction(head, student, intra)
    pixels = restore_rgb(crop_center(rec, rh, rw).float(), rh, rw)
    mse = (pixels - rgb).square().mean(dim=(1, 2, 3))
    perceptual = lpips(pixels * 2 - 1, rgb * 2 - 1).flatten()
    weights = config["distortion_weights"]
    rgb_distortion = weights["rgb_mse"] * mse + weights["lpips_alex"] * perceptual
    by_prefix = rgb_distortion.reshape(-1, len(examples))
    monotonic = torch.relu(by_prefix[1:] - by_prefix[:-1]).mean()
    diff = crop_center(student, rh // 8, rw // 8).float() - crop_center(teacher, rh // 8, rw // 8).float()
    teacher_loss = diff.abs().mean() / teacher.float().abs().mean().clamp_min(0.01)
    total = rgb_distortion.mean() + weights["teacher_feature"] * teacher_loss + weights["soft_monotonic"] * monotonic
    return total, torch.stack((total.detach(), mse.mean().detach(), perceptual.mean().detach(),
                               teacher_loss.detach(), monotonic.detach()))


def step_batch(refiner, decoder, lpips, optimizer, examples, crop, config):
    optimizer.zero_grad(set_to_none=True)
    parts = []
    for intra in (False, True):
        group = [example for example in examples if example.intra == intra]
        if not group:
            continue
        loss, metrics = loss_for_frames(refiner, decoder, lpips, group, crop, config)
        weight = len(group) / len(examples)
        (loss * weight).backward()
        parts.append(metrics * weight)
    grad = nn.utils.clip_grad_norm_(refiner.parameters(), config["optimizer"]["clip_grad"], error_if_nonfinite=True)
    if float(grad) == 0:
        raise RuntimeError("no effective refinement gradient")
    optimizer.step()
    recorded = torch.stack(parts).sum(dim=0).tolist()
    return dict(zip(("loss", "rgb_mse", "lpips", "feature", "monotonic"), recorded, strict=True)) | {
        "gradient_norm": grad.item(), "actual_frame_bytes_mean":
        sum(sum(ex.byte_counts) for ex in examples) / len(examples) / len(examples[0].byte_counts)}


def checkpoint(refiner, quantizer, optimizer, args, step, signature):
    path = args.checkpoints / f"d1-seed{args.seed}-step{step}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"refiner": refiner.state_dict(), "quantizer": quantizer.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "signature": signature,
                "cuda_rng_state": torch.cuda.get_rng_state(), "cpu_rng_state": torch.get_rng_state()}, tmp)
    tmp.replace(path)
    return path


def resume(path, refiner, quantizer, optimizer, signature):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["signature"] != signature:
        raise ValueError("resume config, data, source, or seed differs")
    refiner.load_state_dict(state["refiner"])
    quantizer.load_state_dict(state["quantizer"])
    optimizer.load_state_dict(state["optimizer"])
    torch.cuda.set_rng_state(state["cuda_rng_state"])
    torch.set_rng_state(state["cpu_rng_state"])
    return int(state["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(1, 500, 2000), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup(args.seed)
    config = json.loads(CONFIG.read_text())
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    quantizer = UnifiedRefinement().cuda().eval().requires_grad_(False)
    refiner = SharedRefinement().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=config["optimizer"]["lr"],
                                  weight_decay=config["optimizer"]["weight_decay"])
    signature = {"config": sha256_file(CONFIG), "trainer": sha256_file(Path(__file__)),
                 "manifests": reader.hashes, "seed": args.seed,
                 "assets": [sha256_file(assets.intra), sha256_file(assets.ld)]}
    first = resume(args.resume, refiner, quantizer, optimizer, signature) if args.resume else 0
    if args.steps <= first or (args.steps == 2000 and first < 500):
        raise ValueError("P1b requires the matching P1a checkpoint")
    qps = tuple(config["base"][key] for key in ("qp_i", "qp_p"))
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    decoder, cache = collect_examples(assets, quantizer, reader, plans["fit"], config["budgets"], qps)
    decoder.intra.requires_grad_(False)
    decoder._video().requires_grad_(False)
    lpips = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)
    cache_seconds = perf_counter() - begin
    steps = []
    unroll, start = config["unroll"]["2000" if args.steps == 2000 else "500"], perf_counter()
    for index in range(first, args.steps):
        rng = random.Random((args.seed << 24) + index)
        clip = rng.choice(cache)
        offset = 0 if index % 4 == 0 else rng.randrange(0, len(clip) - unroll + 1)
        frames = clip[offset:offset + unroll]
        crop = Crop.sample(rng, config["rgb_crop_hw"], config["latent_guard_cells"])
        tick = perf_counter()
        metrics = step_batch(refiner, decoder, lpips, optimizer, frames, crop, config)
        done = index + 1
        steps.append(metrics | {"step": done, "step_s": perf_counter() - tick, "sequence_length": len(frames)})
        if done % 25 == 0 or done == args.steps:
            print("step", done, "loss", round(metrics["loss"], 6), flush=True)
        if done % 500 == 0 or done == args.steps:
            path = checkpoint(refiner, quantizer, optimizer, args, done, signature)
            write_csv(args.output / f"steps-{first + 1}-{done}.csv", steps)
            write_json(args.output / f"step-{done}.json", {
                "seed": args.seed, "actual_optimizer_steps": done, "added": done - first,
                "optimizer_step_counts": sorted({int(state["step"]) for state in optimizer.state.values()}),
                "cache_seconds": cache_seconds, "fit_seconds": perf_counter() - start,
                "gpu_hours": (perf_counter() - begin) / 3600,
                "train_frame_instances": unroll * (done - first),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "read_counts": reader.counts, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
                "resumed_from": str(args.resume) if args.resume else None, "signature": signature,
                "quantizer_and_probability": "frozen for this pilot; no proxy gradients claimed",
                "rate": "training byte ledgers are physical; loss varies only in reconstruction",
                "stages": {"unroll": unroll, "records": "E0; dependency-ranked; one maximum family per fit frame"},
            })


if __name__ == "__main__":
    main()
