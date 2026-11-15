"""Measure actual official native I/P encode, SPS/IP bytes, independent decode and RGB output."""

import argparse

import torch

from cp_adaptok.backbones.native_baseline import NativeBaseline
from cp_adaptok.cli.eval import native_frame
from cp_adaptok.data import read_video
from cp_adaptok.metrics import MetricSuite, summarize_latency
from cp_adaptok.runtime import metadata, preflight, read_config, resolve, seed_everything, write_json


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrained_s1.yaml")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", default="results/pretrained_native_benchmark")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")
    return parser, args


@torch.no_grad()
def main():
    parser, args = options()
    config = read_config(args.config)
    preflight(config, "baseline")
    device = config["device"]
    seed_everything(config["seed"], device, config["gpu_memory_fraction"])
    spec = dict(config["data"]["test"][0])
    count = max(args.frames, args.warmup)
    if count > spec["frames"]:
        parser.error("requested frames exceed the declared source span")
    spec["frames"] = count
    clip = read_video(spec).to(device)
    paths = config["paths"]
    model = NativeBaseline(
        resolve(paths["ld_checkpoint"]), resolve(paths["intra_checkpoint"]), device, config["model"]["qp"]
    )
    for rgb in clip[: args.warmup]:
        model.round_trip(rgb[None])
    model.reset()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    suite = MetricSuite(device, ("psnr",))
    rows = [native_frame(model, suite, rgb[None], spec) for rgb in clip[: args.frames]]
    report = metadata(config, "native_pretrained")
    report.update(
        real_entropy_bitstream=True,
        coverage_instrumented=False,
        warmup=args.warmup,
        records=rows,
        overall=summarize_latency([row["latency_ms"] for row in rows]),
        steady_inter=summarize_latency([row["latency_ms"] for row in rows if row["frame_type"] == "P"]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(device)),
        timing="RGB conversion + native encode + SPS/IP parsing + independent native decode + RGB output; metrics, source I/O and initialization excluded",
    )
    write_json(resolve(args.output) / "benchmark.json", report)
    print({key: report[key] for key in ("overall", "steady_inter", "warmup")})


if __name__ == "__main__":
    main()
