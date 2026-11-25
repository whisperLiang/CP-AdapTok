"""Train the Anchor on frozen, contiguous Train windows with native base caches."""

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn

from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint
from hla_tok.training.losses import Losses
from hla_tok.uf.anchor import AnchorModel
from hla_tok.uf.base import schedule
from hla_tok.uf.official import LOCK, ROOT, NativeDecoder, NativeEncoder, OfficialAssets, sha256_file
from hla_tok.uf.training import BASE_WEIGHT, LAMBDA, LPIPS_WEIGHT, MONOTONIC_WEIGHT, WEIGHTS, prefix_objective
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from scripts.hla_tok.temporal.inputs import Clip, source_frame
from scripts.hla_tok.uf.anchor_eval import locked_qp
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import csv_at, json_at, setup

CROP = 256
ACCUMULATION = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class TrainingClip:
    rgb: Tensor
    base: Tensor
    base_bytes: list[int]

    def sample(self, start: int, length: int) -> tuple[Tensor, Tensor, list[int]]:
        offset = slice(start, start + length)
        return (self.rgb[offset].cuda().float().contiguous(memory_format=torch.channels_last),
                self.base[offset].cuda().float().contiguous(memory_format=torch.channels_last), self.base_bytes[offset])


@torch.no_grad()
def cache_train(assets: OfficialAssets, reader: AuditedFrames, clips: list[Clip], qp: tuple[int, int],
                seed: int) -> tuple[list[TrainingClip], list[dict]]:
    generator = random.Random(seed)
    encoder, decoder = NativeEncoder(assets), NativeDecoder(assets)
    cache, locations = [], []
    for clip in clips:
        if clip.split != "train":
            raise PermissionError("Anchor gradients are restricted to Train")
        encoder.reset()
        decoder.reset()
        top = generator.randrange((1080 - CROP) // 8 + 1) * 8
        left = generator.randrange((1920 - CROP) // 8 + 1) * 8
        crop = (..., slice(top, top + CROP), slice(left, left + CROP))
        rgb_frames, base_frames, byte_counts = [], [], []
        for index in range(clip.length):
            rgb = source_frame(reader, clip, index)
            intra, reset = schedule(index)
            base = encoder.encode(rgb, index, qp[0 if intra else 1], intra, reset)
            recon = decoder.decode(base)
            rgb_frames.append(rgb[crop].to(device="cpu", dtype=torch.float16))
            base_frames.append(recon[crop].to(device="cpu", dtype=torch.float16))
            byte_counts.append(len(base.payload) + APP_BASE_OVERHEAD)
        cache.append(TrainingClip(torch.cat(rgb_frames), torch.cat(base_frames), byte_counts))
        locations.append({"video": clip.video, "start": clip.start, "frames": clip.length, "top": top, "left": left})
    return cache, locations


def training_update(model: AnchorModel, optimizer: torch.optim.Optimizer, cache: list[TrainingClip],
                    generator: torch.Generator, unroll: int, lpips: nn.Module | None) -> dict:
    optimizer.zero_grad(set_to_none=True)
    rates, distortion, values = [], [], []
    for _ in range(ACCUMULATION):
        clip = cache[int(torch.randint(len(cache), (), generator=generator))]
        start = int(torch.randint(len(clip.rgb) - unroll + 1, (), generator=generator))
        rgb, base, counts = clip.sample(start, unroll)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, terms = prefix_objective(model, rgb, base, counts, lpips)
        (loss / ACCUMULATION).backward()
        values.append(float(loss.detach()))
        rates.append(terms["estimated_bpp"].mean(dim=1))
        distortion.append(terms["distortion"].mean(dim=1))
    gradient = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    if gradient <= 0 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("optimizer received zero or nonfinite gradients/loss")
    optimizer.step()
    return {"loss": sum(values) / ACCUMULATION, "gradient_norm": gradient} | {
        f"{metric}_{budget}": value for metric, rows in (("estimated_bpp", rates), ("distortion", distortion))
        for budget, value in zip((5, 10, 20, 40), torch.stack(rows).mean(dim=0).tolist(), strict=True)
    }


def _identity(assets: OfficialAssets, plan: dict, dataset: str) -> ResumeIdentity:
    training_sources = [Path(__file__), *(ROOT / "hla_tok/uf").glob("*.py")]
    digest = hashlib.sha256("".join(sha256_file(path) for path in sorted(training_sources)).encode()).hexdigest()
    entry = json_at(LOCK)
    plan["upstream"] = {"commit": entry["commit"], "intra_sha256": sha256_file(assets.intra),
                        "ld_sha256": sha256_file(assets.ld)}
    config = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return ResumeIdentity("uf-anchor", config, dataset, entry["commit"], digest)


def run(assets: OfficialAssets, output: Path, checkpoint_dir: Path, qp: tuple[int, int],
        seed: int, end: int, resume: Path | None) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    setup(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    reader, plans = inputs()
    cache, locations = cache_train(assets, reader, plans["fit"], qp, seed)
    plan = {"seed": seed, "base_qp": qp, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "unroll": {"A1": 4, "A2": 8}, "crop": CROP, "gradient_accumulation": ACCUMULATION,
            "budget_weights": WEIGHTS, "rate_weights": LAMBDA, "lpips_weight": LPIPS_WEIGHT,
            "base_weight": BASE_WEIGHT, "monotonic_weight": MONOTONIC_WEIGHT, "crop_plan": locations}
    identity = _identity(assets, plan, reader.hashes["train"])
    model = AnchorModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    rng = torch.Generator().manual_seed(seed)
    loss = Losses("cuda", lpips_vgg=0, lpips_alex=1.0, dists=0).alex
    previous = 0
    history_file = output / "training.csv"
    history = []
    if resume is not None:
        state = read_checkpoint(resume, identity).restore(model, optimizer, scheduler, rng)
        previous = state["step"]
        history = [row for row in csv_at(history_file) if int(row["step"]) <= previous]
    start = perf_counter()
    for step in range(previous + 1, end + 1):
        unroll = 4 if step <= 500 else 8
        evidence = training_update(model, optimizer, cache, rng, unroll, loss)
        scheduler.step()
        history.append(evidence | {"step": step, "unroll": unroll})
        if step % 50 == 0:
            print(step, evidence["loss"], evidence["gradient_norm"], flush=True)
        if step % 500 == 0:
            path = checkpoint_dir / f"anchor-seed{seed}-step{step}.pt"
            save_checkpoint(path, identity, model, optimizer, scheduler, rng, step=step, data_cursor=step,
                            counters={"configuration": plan, "pixel_reads": reader.counts})
            write_csv(history_file, history)
    write_json(output / f"step{end}.json", {"start_step": previous, "optimizer_steps": end,
               "new_steps": end - previous, "training_seconds": perf_counter() - start,
               "checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "read_counts": reader.counts,
               "configuration": plan, "best_of_test": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--base-lock", type=Path, default=OUTPUT / "baseline/base_lock.json")
    parser.add_argument("--output", type=Path, default=OUTPUT / "anchor/training-seed1")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints/hla_tok/uf")
    parser.add_argument("--steps", type=int, choices=(500, 2000), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps > 500 and args.resume is None:
        parser.error("A2 requires an admitted A1 checkpoint")
    run(OfficialAssets.verify(checkpoint_directory(args.weights)), args.output, args.checkpoints,
        locked_qp(args.base_lock), args.seed, args.steps, args.resume)


if __name__ == "__main__":
    main()
