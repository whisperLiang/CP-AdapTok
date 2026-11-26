"""Replay official FP16 HUF1 endpoints before and after joint training."""

import argparse
import os
import subprocess
from pathlib import Path

import torch

from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import quality_replay, summarize_quality, summarize_timing, timing
from scripts.hla_tok.uf_v3.endpoint import load_endpoint
from scripts.hla_tok.v2.protocol import setup

RD_CUTS = (0, 0.5, 1, 1.3947, 1.395, 2, 3, 3.594, 4, 5, 10, 20, 40)


def other_gpu_processes():
    try:
        report = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                          "--format=csv,noheader,nounits"], text=True)
        return sum(int(pid) != os.getpid() for pid in report.splitlines() if pid.strip().isdigit())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--quality", nargs="+", type=float)
    parser.add_argument("--demand", action="store_true")
    parser.add_argument("--timing-budgets", type=int, nargs="+", choices=tuple(BUDGETS), default=tuple(BUDGETS))
    parser.add_argument("--static", type=Path)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if args.raw and not args.static:
        raise ValueError("Raw fallback is meaningful only with a frozen CDF")
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    clips = plans[args.split]
    codec = load_endpoint(assets, args.checkpoint, static=StaticAsset.load(args.static) if args.static else None,
                          raw=args.raw)
    report = {"scope": "official CUDA FP16; display FP16; refinement FP32",
              "fp32_native_and_bf16_native": "NOT_AVAILABLE",
              "shape": [1, 3, 1080, 1920], "model_tag": codec.tag.hex(), "split": args.split,
              "manifest_hashes": reader.hashes, "checkpoint_sha256": sha256_file(args.checkpoint),
              "runtime": environment(), "backend": "E2" if args.raw else "E1" if args.static else "E0"}
    if args.repetitions:
        bud = tuple(args.timing_budgets)
        torch.cuda.reset_peak_memory_stats()
        gpu_peers_before = other_gpu_processes()
        rows = timing(codec, assets, reader, clips, bud, args.repetitions, args.limit, args.demand)
        write_csv(args.output / "timing.csv", rows)
        report.update({"timing": summarize_timing(rows, bud, args.repetitions),
                       "timing_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                       "timing_mode": "on-demand" if args.demand else "Full-max",
                       "other_gpu_compute_before_after": [gpu_peers_before, other_gpu_processes()],
                       "40_mode_identity": BUDGETS[40] == 166666})
    if args.quality:
        rows = quality_replay(codec, assets, reader, clips, args.quality, args.limit)
        write_csv(args.output / "quality.csv", rows)
        report["quality"] = summarize_quality(rows, args.quality)
        report["base_max"] = max(row["base_bytes"] for row in rows)
    write_json(args.output / "summary.json", report | {"read_counts": reader.counts})


if __name__ == "__main__":
    main()
