"""Synchronized serial RGB-to-RGB timings on one H100, with real wire packets."""

import resource
from collections.abc import Callable
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.continuous import ContinuousReceiver, RankOrder, encode_continuous, truncate
from hla_tok.streaming.continuous_wire import WireDtype, deserialize_continuous_tokens, serialize_cpu
from scripts.hla_tok.continuous.evaluate import _require_pass
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, open_inputs, run_arguments
from scripts.hla_tok.continuous.matrix import compute, representative, validation_frames
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.continuous.receiver import compute_values
from scripts.hla_tok.v2.protocol import ROOT, json_at

T = TypeVar("T")
GPU_POINTS = ("begin", "encoded", "ordered", "converted", "received", "cast", "recovered", "rgb")
TIME_SPANS = {
    "encoder": ("begin", "encoded"),
    "ordering_and_truncation": ("encoded", "ordered"),
    "wire_dtype_conversion": ("ordered", "converted"),
    "decoder_dtype_conversion": ("received", "cast"),
    "recovery_network": ("cast", "recovered"),
    "rgb_decoder": ("recovered", "rgb"),
    "decoder": ("received", "rgb"),
}
WIRE_WALL = ("gpu_to_cpu", "serialization", "deserialization", "cpu_to_gpu")
MIB = 1024**2


def timed(action: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    result = action()
    return result, (perf_counter() - start) * 1000


def latency_summary(values: list[float]) -> dict:
    p50, p95, p99 = (float(value) for value in np.percentile(values, (50, 95, 99)))
    average = float(np.mean(values))
    return {"mean_ms": average, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "fps_from_mean_ms": 1000 / average if average else None}


class TimedPath:
    def __init__(self, inputs: Inputs, order: RankOrder, budget: int, precision: str, dtype: WireDtype | None):
        self.model = inputs.model
        self.receiver = ContinuousReceiver.from_model(inputs.model)
        self.order, self.budget, self.precision, self.dtype = order, budget, precision, dtype
        self.events = {name: torch.cuda.Event(enable_timing=True) for name in GPU_POINTS}
        self.packet_bytes: int | None = None

    def _transfer(self, prefix: Tensor, frame_id: int, dtype: WireDtype) -> tuple[Tensor, dict]:
        converted = prefix.to(dtype.torch_dtype).contiguous()
        self.events["converted"].record()
        self.events["converted"].synchronize()
        cpu, d2h = timed(converted.cpu)
        wire, encode = timed(lambda: serialize_cpu(cpu, dtype, frame_id, order=self.order.name))
        packet, decode = timed(lambda: deserialize_continuous_tokens(wire))
        gpu, h2d = timed(lambda: packet.values.to(prefix.device))
        self.packet_bytes = len(wire)
        return gpu, dict(zip(WIRE_WALL, (d2h, encode, decode, h2d), strict=True))

    def frame(self, source: Tensor, frame_id: int) -> dict:
        e = self.events
        start = perf_counter()
        e["begin"].record()
        native = encode_continuous(self.model, source)
        e["encoded"].record()
        prefix = truncate(self.order.apply(native), self.budget)
        e["ordered"].record()
        received, wall = (prefix, {}) if self.dtype is None else self._transfer(prefix, frame_id, self.dtype)
        e["received"].record()
        received = compute_values(received, self.precision)
        e["cast"].record()
        grid = self.receiver.recover(received, self.order)
        e["recovered"].record()
        self.receiver.reconstruct(grid)
        e["rgb"].record()
        e["rgb"].synchronize()
        total = (perf_counter() - start) * 1000
        stages = {name: e[a].elapsed_time(e[b]) for name, (a, b) in TIME_SPANS.items()
                  if name != "wire_dtype_conversion" or self.dtype is not None}
        return stages | wall | {"total_serial": total}


def _repeat(path: TimedPath, frames: list[Tensor], config: dict) -> tuple[list[dict], float]:
    warmup, count = config["warmup_frames"], config["measured_frames"]
    for index in range(warmup):
        path.frame(frames[index % len(frames)], index)
    start = perf_counter()
    measured = [path.frame(frames[index % len(frames)], index) for index in range(count)]
    elapsed = perf_counter() - start
    return measured, elapsed


def _stage_rows(identity: dict, times: list[dict]) -> list[dict]:
    return [
        identity | {"stage": key, "timing": "wall" if key in (*WIRE_WALL, "total_serial") else "cuda_event"}
        | latency_summary([frame[key] for frame in times])
        for key in times[0]
    ]


def _memory(identity: dict, inputs: Inputs) -> dict:
    params = sum(param.numel() for param in inputs.model.parameters())
    unused = sum(param.numel() for param in inputs.model.quantizer.parameters())
    return identity | {"peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / MIB,
                       "peak_reserved_gpu_mib": torch.cuda.max_memory_reserved() / MIB,
                       "process_cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       "model_parameters": params, "active_continuous_parameters": params - unused}


def _measure(inputs: Inputs, order: RankOrder, frames: list[Tensor], config: dict, identity: dict) -> dict:
    label = identity["wire_dtype"]
    tensor_path = TimedPath(inputs, order, identity["budget"], identity["compute_dtype"],
                            None if label == "none" else WireDtype(label))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    repeat_fps = []
    elapsed_total = 0.0
    with compute(identity["compute_dtype"]):
        for _ in range(config["repeats"]):
            rows, seconds = _repeat(tensor_path, frames, config)
            times.extend(rows)
            elapsed_total += seconds
            repeat_fps.append(len(rows) / seconds)
    breakdown = _stage_rows(identity, times)
    total = latency_summary([row["total_serial"] for row in times])
    stage = {row["stage"]: row["mean_ms"] for row in breakdown}
    steady = len(times) / elapsed_total
    throughput = identity | total | {
        "actual_packet_bytes": tensor_path.packet_bytes, "steady_state_fps": steady,
        "mean_encoder_ms": stage["encoder"], "mean_decoder_ms": stage["decoder"],
        "minimum_30fps_pass": steady >= config["minimum_fps"], "ideal_60fps_pass": steady >= config["ideal_fps"],
        "p95_deadline_pass": total["p95_ms"] <= config["p95_target_ms"],
        **{f"repeat_{i}_fps": value for i, value in enumerate(repeat_fps, 1)},
    }
    return {"latency": breakdown, "throughput": throughput, "memory": _memory(identity, inputs)}


def _identities(inputs: Inputs, order: RankOrder, config: dict) -> list[dict]:
    choices = product(config["compute_dtypes"], (*inputs.config["wire_formats"], "none"), inputs.config["budgets"])
    return [
        {"compute_dtype": precision, "wire_dtype": dtype, "budget": budget, "order": order.name.name,
         "path": "gpu_only" if dtype == "none" else "float_wire_serial",
         "warmup_frames_per_repeat": config["warmup_frames"], "measured_frames_per_repeat": config["measured_frames"],
         "repeats": config["repeats"], "measured_frames": config["repeats"] * config["measured_frames"]}
        for precision, dtype, budget in choices
    ]


def _bind(inputs: Inputs, output: Path, order: RankOrder, config: dict, refs: list[tuple[str, int]]) -> None:
    sources = ("hla_tok/models/codec.py", "hla_tok/models/local.py", "hla_tok/models/spatial.py",
               "hla_tok/streaming/continuous.py", "hla_tok/streaming/continuous_wire.py",
               "scripts/hla_tok/continuous/receiver.py", "scripts/hla_tok/continuous/benchmark.py")
    provenance = inputs.provenance() | {
        "hardware": environment(), "benchmark": config, "order": order.name.name, "ranks": list(order.ranks),
        "sources": [list(ref) for ref in refs], "results_use_repeated_source_frames": True,
        "input_rgb_residency": "GPU normalized native tensor", "output_rgb_residency": "GPU native tensor",
        "disk_io_included": False, "external_network_wait_included": False,
        "implementation": {path: sha256_file(ROOT / path) for path in sources},
    }
    binding = output / "benchmark_manifest.json"
    if binding.exists() and json_at(binding) != provenance:
        raise ValueError("benchmark cache identity differs from the actual evaluation path")
    write_json(binding, provenance)


def _case(inputs: Inputs, output: Path, order: RankOrder, frames: list[Tensor], config: dict, identity: dict) -> dict:
    key = f'{identity["compute_dtype"]}_{identity["wire_dtype"]}_K{identity["budget"]}'
    path = output / "benchmark_cases" / f"{key}.json"
    if path.is_file():
        return json_at(path)
    measured = _measure(inputs, order, frames, config, identity)
    write_json(path, measured)
    row = measured["throughput"]
    print(f'{key}: {row["steady_state_fps"]:.1f} FPS, P95 {row["p95_ms"]:.3f} ms', flush=True)
    return measured


@torch.inference_mode()
def benchmark(v3: Path, output: Path) -> None:
    _require_pass(output, "wire_prefix_audit", "wire_roundtrip_audit")
    for name in ("wire_rate_distortion.csv", "equal_rate_comparison.csv", "bandwidth_constrained_quality.csv"):
        if not (output / name).is_file():
            raise ValueError("finish the wire rate-distortion phases before benchmarking")
    inputs = open_inputs(v3)
    optimum = json_at(output / "ordering_decision.json")["primary_order"]
    order, = [item for item in load_orders(output) if item.name.name == optimum]
    config = inputs.config["benchmark"]
    refs = representative(inputs.cohort)
    _bind(inputs, output, order, config, refs)
    clip = [source for _, _, source in validation_frames(inputs, refs)]
    cases: list[dict] = []
    for identity in _identities(inputs, order, config):
        cases.append(_case(inputs, output, order, clip, config, identity))
        write_csv(output / "latency_breakdown.csv", [row for case in cases for row in case["latency"]])
        write_csv(output / "throughput_benchmark.csv", [case["throughput"] for case in cases])
        write_csv(output / "memory_benchmark.csv", [case["memory"] for case in cases])
    inputs.finish(output / "benchmark_run.json")


def main() -> None:
    parser = run_arguments(__doc__)
    args = parser.parse_args()
    with locked_output(args.output):
        benchmark(args.v3.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
