"""Resume a validated 8,000-step joint endpoint with 16-frame closed-loop windows."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.native_weights import NativeAdjustments
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.train import Crop
from scripts.hla_tok.uf_v2.train_joint import (
    CONFIG,
    NativeProxy,
    activate,
    cache_closed_loop,
    model_and_state,
    step,
    write_checkpoint,
)
from scripts.hla_tok.v2.protocol import ROOT, setup

FORMAL = ROOT / "configs/hla_tok/uf_v2/formal.json"


def signatures(args, assets, hashes):
    p2 = {"config": sha256_file(CONFIG), "source": sha256_file(Path(__file__).with_name("train_joint.py")),
          "seed": args.seed, "manifests": hashes, "weights": [sha256_file(p) for p in (assets.intra, assets.ld)]}
    formal = p2 | {"formal_config": sha256_file(FORMAL), "formal_source": sha256_file(Path(__file__))}
    return p2, formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup(args.seed)
    config = json.loads(CONFIG.read_text())
    gate = json.loads(FORMAL.read_text())
    asset = OfficialAssets.verify(args.weights)
    reader, plans = inputs()
    p2, formal = signatures(args, asset, reader.hashes)
    state = torch.load(args.resume, map_location="cpu", weights_only=True)
    origin = int(state["step"])
    if origin < gate["initial_step"] or args.steps != origin + gate["evaluation_interval"]:
        raise ValueError("formal fitting requires the previous evaluated checkpoint and exactly one new node")
    if args.steps > gate["maximum_step"]:
        raise ValueError("formal cumulative optimizer budget exceeded")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    modules, optimizer, first = model_and_state(args, config, p2 if origin == gate["initial_step"] else formal)
    refiner, quantizer, adjustments = modules
    encoded = NativeAdjustments().cuda().requires_grad_(False)
    encoded.load_state_dict(adjustments.state_dict())
    metric = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    decoder, cache = cache_closed_loop(asset, quantizer, refiner, encoded, reader, plans["fit"])
    native = NativeProxy(decoder, adjustments, encoded)
    unroll = gate["unroll"]
    cache_time = perf_counter() - start
    trace = []
    for i in range(first, args.steps):
        activate(adjustments, i)
        rng = random.Random((args.seed << 24) + i)
        video = rng.choice(cache)
        offset = 0 if i % 4 == 0 else rng.randrange(0, len(video) - unroll + 1)
        crop = Crop.sample(rng, config["crop_hw"], config["guard_latent_cells"])
        tick = perf_counter()
        stats = step(modules, native, metric, optimizer, video[offset:offset + unroll], crop, config)
        trace.append(stats | {"step": i + 1, "unroll": unroll, "step_s": perf_counter() - tick})
        if (i + 1) % 100 == 0:
            print(i + 1, round(stats["loss"], 6), flush=True)
    path = write_checkpoint(args, modules, optimizer, formal, args.steps)
    write_csv(args.output / f"steps-{first + 1}-{args.steps}.csv", trace)
    write_json(args.output / f"step-{args.steps}.json", {
        "checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "optimizer_steps": args.steps,
        "added_steps": args.steps - first, "seed": args.seed, "unroll": unroll,
        "resume": str(args.resume), "signature": formal, "train_frame_instances": unroll * (args.steps - first),
        "cache_seconds": cache_time, "read_counts": reader.counts,
        "gpu_hours": (perf_counter() - start) / 3600, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "mean_step_s": sum(r["step_s"] for r in trace) / len(trace),
    })


if __name__ == "__main__":
    main()
