"""Synchronized CPU wall/CUDA event breakdown; run separate uninstrumented throughput tests."""

import argparse
from collections import defaultdict

import numpy as np

from cp_adaptok.evaluation import benchmark_pipeline
from cp_adaptok.profiling import PipelineProfiler
from cp_adaptok.runtime import build_model, metadata, read_config, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 2048])
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=args.checkpoint)
    model.eval()
    benchmark_pipeline(model, config, [1920, 1080], 2, 0, args.budgets)
    profiler = PipelineProfiler(model.device)
    model.pipeline_profiler = profiler
    benchmark = benchmark_pipeline(model, config, [1920, 1080], args.frames, 0, args.budgets)
    grouped = defaultdict(list)
    for row in profiler.records:
        grouped[row["name"]].append(row)
    summaries = {
        name: {
            "calls": len(rows),
            "wall_mean_ms": float(np.mean([r["wall_ms"] for r in rows])),
            "cuda_mean_ms": float(np.mean([r["cuda_ms"] for r in rows])),
        }
        for name, rows in grouped.items()
    }
    write_json(
        args.output,
        {
            **metadata(config, "synchronized_profile"),
            "checkpoint": args.checkpoint,
            "instrumented_not_throughput": True,
            "nested_timings_not_additive": True,
            "breakdown": summaries,
            "events": profiler.records,
            "pipeline": benchmark,
        },
    )
    print(summaries)


if __name__ == "__main__":
    main()
