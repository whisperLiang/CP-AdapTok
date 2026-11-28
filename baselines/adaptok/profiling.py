"""Synchronized wall time and CUDA events; all rates count reconstructed source frames."""

import time

import numpy as np
import torch


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def measured(call, device):
    synchronize(device)
    cuda = torch.device(device).type == "cuda"
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if cuda else None
    begin = time.perf_counter()
    if cuda:
        events[0].record()
    value = call()
    if cuda:
        events[1].record()
    synchronize(device)
    wall_ms = (time.perf_counter() - begin) * 1000
    return value, {"wall_ms": wall_ms, "cuda_ms": events[0].elapsed_time(events[1]) if cuda else None}


def latency_summary(windows):
    result = {"windows": windows, "source_frames": sum(w["valid_frames"] for w in windows)}
    for name in ["encode_ms", "decode_ms", "tiling_ms", "stitch_ms", "compute_ms", "read_ms", "write_ms", "metrics_ms"]:
        values = np.array([w[name] for w in windows], dtype=float)
        result[name] = {"total": float(values.sum()), "mean": float(values.mean()),
                        "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
    for label, names in {"neural": ["encode_ms", "decode_ms"], "compute": ["compute_ms"],
                         "pipeline": ["read_ms", "compute_ms", "write_ms"],
                         "neural_cuda_events": ["encode_cuda_ms", "decode_cuda_ms"]}.items():
        durations = np.array([sum(w[k] for k in names) for w in windows])
        per_frame = np.array([d / w["valid_frames"] for d, w in zip(durations, windows)])
        total = float(durations.sum())
        result[label] = {"total_ms": total, "total_fps": result["source_frames"] * 1000 / total if total > 0 else None,
                         "mean_ms_per_frame": total / result["source_frames"],
                         "p50_ms_per_frame": float(np.percentile(per_frame, 50)),
                         "p95_ms_per_frame": float(np.percentile(per_frame, 95))}
    result["metrics_excluded_from_all_fps"] = True
    return result
