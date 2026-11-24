"""Synchronized native RGB benchmark for the stateless T0 architecture."""

from collections import defaultdict
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from hla_tok import Geometry, HLATok
from hla_tok.evaluation.io import timed_cuda, write_csv, write_json
from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit

BENCH_BUDGETS = (4096, 8192, 16320, 32640, 65280)
NATIVE = Geometry(1080, 1920)


def transfer(rgb):
    return rgb.cuda(non_blocking=True).to(dtype=torch.float32, memory_format=torch.channels_last).div_(255)


def codec_call(model, frame, budget, transfer_rgb=False):
    with record_function("hla/host_device"):
        frame = transfer(frame) if transfer_rgb else frame
    ids = model.encode_frame(frame, budget)
    with record_function("hla/fixed13_and_id_transfer"):
        payload = pack_tokens_13bit(ids[0].cpu().tolist())
        received = torch.tensor([unpack_tokens_13bit(payload, budget)], dtype=torch.long, device=frame.device)
    reconstructed = model.decode_prefix(received, NATIVE)
    if transfer_rgb:
        with record_function("hla/host_device"):
            reconstructed = reconstructed.cpu()
    return reconstructed


def benchmark_latency(model, clip, device_clip, budget, repeats):
    measurements = defaultdict(list)
    for _ in range(repeats):
        for raw, frame in zip(clip, device_clip):
            frame, raw = frame[None], raw[None]
            t0, t1, t2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            torch.cuda.synchronize()
            began = perf_counter()
            t0.record()
            ids = model.encode_frame(frame, budget)
            t1.record()
            model.decode_prefix(ids, NATIVE)
            t2.record()
            t2.synchronize()
            measurements["neural_wall_ms"].append((perf_counter() - began) * 1000)
            measurements["encoder_cuda_ms"].append(t0.elapsed_time(t1))
            measurements["decoder_cuda_ms"].append(t1.elapsed_time(t2))
            measurements["neural_cuda_ms"].append(t0.elapsed_time(t2))
            _, wall, cuda = timed_cuda(lambda: codec_call(model, frame, budget))
            measurements["fixed13_wall_ms"].append(wall)
            measurements["fixed13_cuda_ms"].append(cuda)
            _, wall, cuda = timed_cuda(lambda: codec_call(model, raw, budget, transfer_rgb=True))
            measurements["io_wall_ms"].append(wall)
            measurements["io_cuda_ms"].append(cuda)
    return {key: mean(rows) for key, rows in measurements.items()} | {
        f"{key}_p{p}": float(np.percentile(measurements[key], p))
        for key in ("neural_wall_ms", "fixed13_wall_ms", "io_wall_ms") for p in (50, 95)
    }


def component_times(model, raw, budget):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as traced:
        codec_call(model, raw, budget, transfer_rgb=True)
    averages = traced.key_averages()
    annotations = [event for event in averages if event.key.startswith("hla/")]
    gpu_spans = {event.key: event.device_time_total / 1000 for event in annotations
                 if event.device_type == torch.autograd.DeviceType.CUDA}
    stages = [{"stage": event.key.removeprefix("hla/"), "calls": event.count,
               "cpu_ms": event.cpu_time_total / 1000, "cuda_kernel_ms": event.device_time_total / 1000,
               "cuda_span_ms": gpu_spans.get(event.key)}
              for event in annotations if event.device_type == torch.autograd.DeviceType.CPU]
    kernels = [event.key for event in averages if event.key.startswith("aten::_scaled_dot_product")]
    return stages, sum(event.flops for event in averages), kernels


@torch.inference_mode()
def native_benchmark(clip, output, env, *, budgets=BENCH_BUDGETS, repeats=2):
    torch.manual_seed(1)
    model = HLATok().eval().cuda().to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    device_clip = transfer(clip)
    rows, components = [], []
    for precision in ("bf16", "fp32"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            for budget in budgets:
                _, first_ms, _ = timed_cuda(lambda: model.decode_prefix(model.encode_frame(device_clip[:1], budget), NATIVE))
                for frame in device_clip[:4]:
                    model.decode_prefix(model.encode_frame(frame[None], budget), NATIVE)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated() / 2**20
                latency = benchmark_latency(model, clip, device_clip, budget, repeats)
                peak_allocated = torch.cuda.max_memory_allocated() / 2**20
                peak_reserved = torch.cuda.max_memory_reserved() / 2**20
                stages, flops, kernels = component_times(model, clip[:1], budget)
                label = {"precision": precision, "budget": budget}
                components += [label | stage for stage in stages]
                rows.append(label | {
                    "is_random_weight_architecture_test": True, "batch": 1, "source_frames": len(clip),
                    "repeats": repeats, "parameters": params, "measurable_flops": flops,
                    "first_ms": first_ms, "compile_s": 0, "peak_allocated_mib": peak_allocated,
                    "peak_reserved_mib": peak_reserved, "preloaded_input_and_model_mib": baseline,
                    "encoder_cuda_fps": 1000 / latency["encoder_cuda_ms"],
                    "decoder_cuda_fps": 1000 / latency["decoder_cuda_ms"],
                    "neural_wall_fps": 1000 / latency["neural_wall_ms"],
                    "io_serial_fps": 1000 / latency["io_wall_ms"],
                    "attention_kernels": ",".join(kernels),
                } | latency)
                print(f"T0 {precision} K{budget} {rows[-1]['neural_wall_fps']:.2f} serial neural FPS", flush=True)
                write_csv(output / "benchmark.csv", rows)
                write_csv(output / "components.csv", components)
    write_json(output / "environment.json", env | {"random_seed": 1, "geometry": [1080, 1920],
               "fixed13_rate_scope": "benchmark transports only standalone index payloads",
               "io": "serial raw RGB host-to-device and reconstruction device-to-host; no disk"})
    return rows


def _prefetch(reader, references):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        frames = iter(references)
        pending = pool.submit(reader.read, "validation", *next(frames))
        for reference in frames:
            frame = pending.result()
            pending = pool.submit(reader.read, "validation", *reference)
            yield frame
        yield pending.result()


@torch.inference_mode()
def native_pipeline(reader, references, output, *, budgets=BENCH_BUDGETS, repeats=2):
    torch.manual_seed(1)
    model = HLATok().cuda().eval().to(memory_format=torch.channels_last)
    rows = []
    for precision in ("bf16", "fp32"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            raw = reader.read("validation", *references[0])
            for budget in budgets:
                codec_call(model, raw[None], budget, transfer_rgb=True)
                torch.cuda.synchronize()
                for repetition in range(repeats):
                    started = perf_counter()
                    for frame in _prefetch(reader, references):
                        codec_call(model, frame[None], budget, transfer_rgb=True)
                    torch.cuda.synchronize()
                    seconds = perf_counter() - started
                    rows.append({"budget": budget, "precision": precision, "repetition": repetition + 1,
                                 "source_frames": len(references), "wall_seconds": seconds,
                                 "pipeline_fps": len(references) / seconds,
                                 "io_scope": "verified source-file read and RGB decode, one prefetch worker, RGB out to CPU"})
                write_csv(output / "pipeline.csv", rows)
    return rows
