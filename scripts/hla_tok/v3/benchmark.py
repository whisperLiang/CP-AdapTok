"""Synchronized timing for trained native 1080p reconstruction."""

from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from cp_adaptok.metrics import summarize_latency
from hla_tok.evaluation.benchmark import _prefetch, transfer
from hla_tok.evaluation.io import timed_cuda, write_csv, write_json
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger
from scripts.hla_tok.v3.protocol import NATIVE


def native_references(cohort, count):
    refs = [(video, index) for video, frames in cohort.items() for index in range(frames)][:count]
    if len(refs) != count:
        raise ValueError("frozen Validation does not supply the requested number of real frames")
    return refs


def _conv_macs(model, source):
    macs = 0

    def count(module, inputs, output):
        nonlocal macs
        height, width = module.kernel_size
        macs += output.numel() * (module.in_channels // module.groups) * height * width

    hooks = [module.register_forward_hook(count) for module in model.modules() if isinstance(module, nn.Conv2d)]
    try:
        model(source)
    finally:
        for hook in hooks:
            hook.remove()
    return macs


@torch.inference_mode()
def dense_benchmark(model, reader, cohort, config, output):
    model.eval()
    controls = config["performance"]
    references = native_references(cohort, controls["frames"])
    before = dict(reader.counts)
    rows = []
    try:
        clip = transfer(torch.stack([reader.read("validation", *ref) for ref in references]).pin_memory())
        conv_macs = _conv_macs(model, clip[:1])
        for precision in ("bf16", "fp32"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                for frame in clip[:4]:
                    model(frame[None])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                durations = []
                for _ in range(controls["repeats"]):
                    for frame in clip:
                        start = perf_counter()
                        model(frame[None])
                        torch.cuda.synchronize()
                        durations.append((perf_counter() - start) * 1e3)
                peak = torch.cuda.max_memory_allocated() / 2**20
                reserved = torch.cuda.max_memory_reserved() / 2**20
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as trace:
                    model(clip[:1])
                rows.append({
                    "mode": "dense", "precision": precision, "frames": len(references),
                    "repeats": controls["repeats"], "physical_batch": 1,
                    "parameters": sum(param.numel() for param in model.parameters()),
                    "measurable_flops": sum(event.flops for event in trace.key_averages()),
                    "conv_only_macs": conv_macs, "conv_only_flops": 2 * conv_macs,
                    "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
                    "p99_ms": float(torch.tensor(durations).quantile(0.99)),
                } | summarize_latency(durations))
        write_csv(output / "timings.csv", rows)
        return rows
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})


def _wire(model, receiver, source, budget, *, copy_rgb=False):
    if copy_rgb:
        with record_function("hla/host_device_rgb"):
            source = transfer(source)
    ids = model.encode_frame(source, budget)
    with record_function("hla/host_device_ids"):
        raw = ids[0].cpu().tolist()
    with record_function("hla/fixed13_serialize"):
        payload = pack_tokens_13bit(raw)
    pixels = receiver.decode_bytes(payload, budget, NATIVE)
    if copy_rgb:
        with record_function("hla/host_device_rgb"):
            pixels = pixels.cpu()
    return pixels, len(payload)


def _profile_codec(model, receiver, source, budget):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as trace:
        _wire(model, receiver, source, budget, copy_rgb=True)
    events = trace.key_averages()
    stages = [{"stage": event.key.removeprefix("hla/"), "calls": event.count,
               "cpu_ms": event.cpu_time_total / 1e3, "cuda_kernel_ms": event.device_time_total / 1e3}
              for event in events if event.key.startswith("hla/")
              and event.device_type == torch.autograd.DeviceType.CPU]
    return sum(event.flops for event in events), stages


def _measure_pipeline(model, receiver, reader, refs, budget, repeats):
    elapsed, per_frame = [], []
    for _ in range(repeats):
        start = perf_counter()
        frames = iter(_prefetch(reader, refs))
        for _ in refs:
            frame_start = perf_counter()
            raw = next(frames)
            _wire(model, receiver, raw[None], budget, copy_rgb=True)
            per_frame.append((perf_counter() - frame_start) * 1e3)
        frames.close()
        torch.cuda.synchronize()
        elapsed.append(perf_counter() - start)
    return _latency_fields("pipeline", per_frame) | {
        "pipeline_fps": len(refs) * repeats / sum(elapsed),
        "pipeline_repeat_fps": ";".join(str(len(refs) / seconds) for seconds in elapsed),
    }


def _latency_fields(name, rows):
    stats = summarize_latency(rows)
    stats["p99_ms"] = float(np.percentile(rows, 99))
    return {f"{name}_{key}": value for key, value in stats.items()}


def _serial_pass(model, receiver, reader, refs, clip, budget, repeats):
    durations: dict[str, list[float]] = {key: [] for key in ("encoder", "decoder", "neural", "fixed13", "serial")}
    byte_count = 0
    for _ in range(repeats):
        for ref, frame in zip(refs, clip, strict=True):
            rgb = frame[None]
            ids, encode_wall, _ = timed_cuda(lambda: model.encode_frame(rgb, budget))
            _, decode_wall, _ = timed_cuda(lambda: receiver.decode_prefix(ids, NATIVE))
            durations["encoder"].append(encode_wall)
            durations["decoder"].append(decode_wall)
            _, neural_wall, _ = timed_cuda(lambda: receiver.decode_prefix(model.encode_frame(rgb, budget), NATIVE))
            durations["neural"].append(neural_wall)
            (_, byte_count), fixed_wall, _ = timed_cuda(lambda: _wire(model, receiver, rgb, budget))
            durations["fixed13"].append(fixed_wall)
            start = perf_counter()
            raw = reader.read("validation", *ref)
            _wire(model, receiver, raw[None], budget, copy_rgb=True)
            torch.cuda.synchronize()
            durations["serial"].append((perf_counter() - start) * 1e3)
    row = {key: value for name, values in durations.items() for key, value in _latency_fields(name, values).items()}
    deadline = 1000 / 30
    row["deadline_ms"] = deadline
    row["serial_deadline_miss_fraction"] = sum(value > deadline for value in durations["serial"]) / len(
        durations["serial"])
    row["serial_p95_meets_30fps_deadline"] = row["serial_p95_ms"] <= deadline
    return row | payload_ledger(byte_count, budget, 1, NATIVE.height, NATIVE.width)


@torch.inference_mode()
def codec_benchmark(model, reader, cohort, config, output, budgets, *, precisions=("bf16", "fp32")):
    model.eval()
    receiver = model.receiver().eval()
    controls = config["performance"]
    refs = native_references(cohort, controls["frames"])
    before = dict(reader.counts)
    rows, components = [], []
    try:
        raw = torch.stack([reader.read("validation", *ref) for ref in refs]).pin_memory()
        clip = transfer(raw)
        for precision in precisions:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                for budget in budgets:
                    for frame in clip[:4]:
                        _wire(model, receiver, frame[None], budget)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    durations = _serial_pass(model, receiver, reader, refs, clip, budget, controls["repeats"])
                    throughput = _measure_pipeline(model, receiver, reader, refs, budget, controls["repeats"])
                    peak = torch.cuda.max_memory_allocated() / 2**20
                    reserved = torch.cuda.max_memory_reserved() / 2**20
                    flops, parts = _profile_codec(model, receiver, raw[:1], budget)
                    labels = {"budget": budget, "precision": precision, "repeats": controls["repeats"],
                              "frames": len(refs), "physical_batch": 1}
                    rows.append(labels | durations | throughput | {
                        "peak_allocated_mib": peak, "peak_reserved_mib": reserved, "measurable_flops": flops,
                        "measurable_macs": flops / 2, "parameters": sum(param.numel() for param in model.parameters()),
                    })
                    components.extend(labels | part for part in parts)
                    write_csv(output / "timings.csv", rows)
                    write_csv(output / "components.csv", components)
                    print(f"benchmark {precision} K{budget}: {throughput['pipeline_fps']:.3f} pipeline FPS", flush=True)
        return rows
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
