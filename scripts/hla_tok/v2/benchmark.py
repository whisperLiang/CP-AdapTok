"""Measured trained-weight architecture timing on 64 verified native frames."""

# ruff: noqa: E402
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from hla_tok import HLATok
from hla_tok.evaluation.benchmark import (
    BENCH_BUDGETS,
    NATIVE,
    _prefetch,
    benchmark_latency,
    codec_call,
    component_times,
    transfer,
)
from hla_tok.evaluation.io import environment, timed_cuda, write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.training.v2 import load_experimental_checkpoint
from scripts.hla_tok.v2.protocol import COHORT, OUTPUT, ROOT, audit_read_counts, frozen_data, json_at, setup


def first_refs(cohort):
    return [(video, idx) for video, count in cohort.items() for idx in range(count)][:64]


def cache_sizes(model, frame, budget):
    context = model.prepare_frame(frame)
    model.extend_prefix(context, budget)
    kv = sum(t.numel() * t.element_size() for pair in context.memory for t in pair) if not isinstance(
        context.memory, torch.Tensor) else context.memory.numel() * context.memory.element_size()
    return {"sender_spatial_cache_bytes": kv,
            "sender_rank_cache_bytes": 0 if context.rank_history is None else context.rank_history.nbytes}


def pipeline(model, reader, refs, budget, repeats):
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        for frame in _prefetch(reader, refs):
            codec_call(model, frame[None], budget, transfer_rgb=True)
        torch.cuda.synchronize()
        durations.append(perf_counter() - start)
    return {"pipeline_fps": len(refs) * repeats / sum(durations), "pipeline_repeat_seconds": ",".join(map(str, durations))}


def times(model, reader, refs, clip, typed, budget, repeats):
    one = typed[:1]
    _, first, _ = timed_cuda(lambda: model.decode_prefix(model.encode_frame(one, budget), NATIVE))
    for image in typed[:4]:
        model.decode_prefix(model.encode_frame(image[None], budget), NATIVE)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated() / 2**20
    neural = benchmark_latency(model, clip, typed, budget, repeats)
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    components, flops, kernels = component_times(model, clip[:1], budget)
    return neural | pipeline(model, reader, refs, budget, repeats) | cache_sizes(model, one, budget) | {
        "neural_fps": 1000 / neural["neural_wall_ms"], "fixed13_fps": 1000 / neural["fixed13_wall_ms"],
        "encoder_cuda_fps": 1000 / neural["encoder_cuda_ms"], "decoder_cuda_fps": 1000 / neural["decoder_cuda_ms"],
        "first_neural_wall_ms": first, "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
        "preloaded_input_and_model_mib": baseline_mem, "measurable_flops": flops,
        "attention_ops": ",".join(kernels), "parameters": sum(p.numel() for p in model.parameters()),
    }, components


@torch.inference_mode()
def measure(model, reader, refs, output, metadata):
    clip = torch.stack([reader.read("validation", video, idx) for video, idx in refs]).pin_memory()
    typed = transfer(clip)
    rows, profiler = [], defaultdict(list)
    budgets = [k for k in BENCH_BUDGETS if k <= model.architecture.local_capacity * NATIVE.num_regions]
    for precision in ("bf16", "fp32"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            for budget in budgets:
                labels = metadata | {"precision": precision, "budget": budget, "source_frames": len(clip),
                                     "repeats": 2, "batch": 1}
                result, components = times(model, reader, refs, clip, typed, budget, 2)
                rows.append(labels | result)
                profiler["components"] += [labels | component for component in components]
                write_csv(output / "timings.csv", rows)
                write_csv(output / "components.csv", profiler["components"])
                print(labels["arm"], precision, budget, round(result["neural_fps"], 2),
                      round(result["pipeline_fps"], 2), flush=True)


def pilot_path(seed, arm):
    directory = OUTPUT / f"continuous_pilot/seed{seed}/{arm}/pilot"
    return ROOT / json_at(directory / "latest.json")["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("R0", "R1", "R2", "R3"), required=True)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    setup(args.seed)
    checkpoint = args.checkpoint or pilot_path(args.seed, args.arm)
    output = args.output or OUTPUT / f"benchmark/seed{args.seed}/{args.arm}"
    output.mkdir(parents=True, exist_ok=False)
    reader, cohort, _ = frozen_data()
    model = HLATok(args.arm).cuda().to(memory_format=torch.channels_last).eval()
    loaded = load_experimental_checkpoint(checkpoint, model, sha256_file(COHORT))
    stage = loaded.state["identity"]["stage"]
    label = {"arm": args.arm, "weights": stage, "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(output / "environment.json", environment() | label | {
        "fps_scope": "neural serial; real Fixed13 and device copy; real source decode + codec pipeline",
        "untrained_rate_points": "Timing only outside the trained fixed budget; continuous arms use uncalibrated VQ for timing.",
    })
    try:
        measure(model, reader, first_refs(cohort), output, label)
    finally:
        audit_read_counts(output, reader)


if __name__ == "__main__":
    main()
