"""Resumable experiments: completed windows are immutable and checked by content/config identity."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .artifacts import concatenate_videos, write_csv, write_json, write_samples, write_video
from .metrics import aggregate, boundary_mask
from .profiling import latency_summary, measured
from .reconstruction import reconstruct_window
from .tiling import TileLayout
from .video_reader import read_input, source_metadata, temporal_windows


def budget_timing(timing, budget, valid_frames, read_ms, write_ms, metrics_ms):
    encode, decode = timing["encode_ms"], timing["decode_ms_by_budget"][budget]
    stitch = timing["stitch_ms_by_budget"][budget]
    return {"valid_frames": valid_frames, "tile_count": timing["tile_count"], "tile_batch_size": timing["tile_batch_size"],
            "read_ms": read_ms, "tiling_ms": timing["tiling_ms"], "encode_ms": encode, "decode_ms": decode,
            "encode_cuda_ms": timing["encode_cuda_ms"], "decode_cuda_ms": timing["decode_cuda_ms_by_budget"][budget],
            "stitch_ms": stitch, "compute_ms": timing["tiling_ms"] + encode + decode + stitch + timing["shared_overhead_ms"],
            "shared_overhead_ms": timing["shared_overhead_ms"], "write_ms": write_ms, "metrics_ms": metrics_ms,
            "all_budgets_compute_wall_ms": timing["all_budgets_wall_ms"],
            "timing_attribution": "single budget measured wall" if len(timing["decode_ms_by_budget"]) == 1 else "component sum; full encoder/read/shared overhead charged independently to each budget"}


def quality_rows(source, window, budget, stride, original, reconstructed, evaluator, mask):
    rows = []
    for local, frame_id in enumerate(window.frame_indices):
        rows.append({"content_id": source["content_id"], "frame_id": frame_id, "stride": stride,
                     "tokens_per_tile_window": budget, "bpp": None, "quality_only": True,
                     **evaluator(original[local:local + 1], reconstructed[local:local + 1], mask)})
    return rows


def window_outputs(directory, source, window, reconstructions, evaluator, layout, config):
    original = window.frames[:window.valid_frames].to(config["device"])
    mask = boundary_mask(layout.height, layout.width, config["boundary_spacing"], config["boundary_half_width"], config["device"])
    _, cost = measured(lambda: write_video(directory / "original.mp4", original, source["fps"]), config["device"])
    original_write_ms = cost["wall_ms"]
    results = {}
    for budget, reconstructed in reconstructions.items():
        reconstructed = reconstructed[:window.valid_frames]
        rows, metric_cost = measured(lambda: quality_rows(source, window, budget, layout.stride, original, reconstructed, evaluator, mask), config["device"])
        def save():
            write_video(directory / f"k{budget}.mp4", reconstructed, source["fps"])
            write_samples(directory / f"samples_k{budget}", original, reconstructed)
        _, write_cost = measured(save, config["device"])
        results[str(budget)] = {"frames": rows, "metrics_ms": metric_cost["wall_ms"],
                                "write_ms": original_write_ms + write_cost["wall_ms"]}
    return results


def run_window(source, adapter, evaluator, directory, start, count, stride, budgets, config):
    clip, metadata = read_input(source["path"], count, start, source["fps"], source.get("raw_yuv"), source)
    window = next(temporal_windows(clip, adapter.frame_num, start))
    layout = TileLayout(1080, 1920, adapter.input_size, stride)
    print(f"{source['content_id']} frames={start}:{start+count} stride={stride} tiles={len(layout.coordinates)} K={budgets}", flush=True)
    torch.cuda.reset_peak_memory_stats(adapter.device)
    reconstructions, timing = reconstruct_window(window.frames, adapter, layout, budgets, config["tile_batch_size"])
    # Record inference peak before full-frame metric networks allocate their activations.
    peak = {"allocated_mib": torch.cuda.max_memory_allocated(adapter.device) / 2**20,
            "reserved_mib": torch.cuda.max_memory_reserved(adapter.device) / 2**20,
            "metric_networks_resident": True}
    output = window_outputs(directory, source, window, reconstructions, evaluator, layout, config)
    for budget in budgets:
        item = output[str(budget)]
        item["latency"] = budget_timing(timing, budget, count, metadata["read_ms"], item["write_ms"], item["metrics_ms"])
    return {"input": metadata, "timing": timing, "peak_inference_memory": peak, "budgets": output,
            "padding_bottom": layout.padded_shape[0] - 1080, "padding_right": layout.padded_shape[1] - 1920,
            "independent_temporal_window": True}


def experiment_identity(source, adapter, stride, budgets, frames, start, config):
    return {"source_sha256": source["sha256"], "content_id": source["content_id"], "revision": adapter.manifest["revision"],
            "weights_sha256": adapter.manifest["model_safetensors_sha256"], "stride": stride, "budgets": budgets,
            "frames": frames, "start": start, "tile_size": adapter.input_size, "frame_num": adapter.frame_num,
            "tile_batch_size": config["tile_batch_size"], "dtype": adapter.manifest["dtype"],
            "boundary_spacing": config["boundary_spacing"], "boundary_half_width": config["boundary_half_width"],
            "raw_yuv": source.get("raw_yuv")}


def prepare_source(record):
    if record.get("raw_yuv"):
        return record
    actual = source_metadata(record["path"], record["fps"])
    if record.get("sha256") and actual["sha256"] != record["sha256"]:
        raise ValueError(f"input content changed since the sealed manifest: {record['content_id']}")
    if (actual["height"], actual["width"]) != (1080, 1920):
        raise ValueError("this experiment requires native 1920x1080; no cropping or resizing")
    return {**record, **actual}


def evaluate_video(record, adapter, evaluator, output, stride, budgets, frames, start, config):
    source = prepare_source(record)
    directory = Path(output) / source["content_id"].replace(":", "_")
    directory.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(source, adapter, stride, budgets, frames, start, config)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    marker = directory / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError(f"refusing to overwrite a different experiment: {directory}")
    write_json(marker, identity)
    if (directory / "completed.json").exists():
        return json.loads((directory / "completed.json").read_text())
    windows = []
    for offset in range(0, frames, adapter.frame_num):
        destination = directory / f"window_{start + offset:06d}"
        completed = destination / "completed.json"
        if completed.exists():
            windows.append(json.loads(completed.read_text()))
            continue
        # Incomplete outputs are isolated under a fresh attempt directory after a crash.
        destination.mkdir(parents=True, exist_ok=True)
        attempt = destination / f"attempt_{time.time_ns()}"
        attempt.mkdir()
        result = run_window(source, adapter, evaluator, attempt, start + offset, min(adapter.frame_num, frames - offset), stride, budgets, config)
        result.update(directory=str(attempt.resolve()), identity=key)
        write_json(completed, result)
        windows.append(result)
    result = finish_video(directory, windows, source, budgets, identity)
    write_json(directory / "completed.json", result)
    return result


def finish_video(directory, windows, source, budgets, identity):
    result = {"identity": identity, "source": source, "windows": windows, "budgets": {}}
    video_dir = directory / "reconstructed_videos"
    video_dir.mkdir(exist_ok=True)
    for budget in budgets:
        items = [w["budgets"][str(budget)] for w in windows]
        rows = [row for item in items for row in item["frames"]]
        begin = time.perf_counter()
        destination = video_dir / f"{source['content_id'].replace(':', '_')}_k{budget}.mp4"
        if not destination.exists():
            concatenate_videos([Path(w["directory"]) / f"k{budget}.mp4" for w in windows], destination)
        items[-1]["latency"]["write_ms"] += (time.perf_counter() - begin) * 1000
        timing = latency_summary([item["latency"] for item in items])
        result["budgets"][str(budget)] = {"metrics": aggregate(rows), "latency": timing, "video": str(destination.resolve())}
        write_csv(directory / f"frame_metrics_k{budget}.csv", rows)
    return result


def summarize_experiment(output, videos, budgets):
    all_rows = [row for video in videos for window in video["windows"] for item in window["budgets"].values() for row in item["frames"]]
    summary = {"quality_only": True, "bpp": None, "budgets": {}, "video_count": len(videos),
               "boundary_definition": "union of [line-2,line+2) pixel strips at internal multiples of 128, same mask for both strides",
               "temporal_state": "each native window independent; repeat last real frame for tail; score only real frames",
               "timing": "single-budget E1/E2 directly measured; multi-budget runs charge shared encoder/read cost to each budget"}
    video_rows = []
    for budget in budgets:
        rows = [r for r in all_rows if r["tokens_per_tile_window"] == budget]
        windows = [w["budgets"][str(budget)]["latency"] for video in videos for w in video["windows"]]
        summary["budgets"][str(budget)] = {"metrics": aggregate(rows), "latency": latency_summary(windows),
            "peak_inference_allocated_mib": max(w["peak_inference_memory"]["allocated_mib"] for v in videos for w in v["windows"]),
            "peak_inference_reserved_mib": max(w["peak_inference_memory"]["reserved_mib"] for v in videos for w in v["windows"])}
        for video in videos:
            video_rows.append({"content_id": video["source"]["content_id"], "tokens_per_tile_window": budget,
                               **video["budgets"][str(budget)]["metrics"]})
    summary["video_mean_metrics"] = {str(k): aggregate_video_means(video_rows, k) for k in budgets}
    write_csv(Path(output) / "frame_metrics.csv", all_rows)
    write_csv(Path(output) / "video_metrics.csv", video_rows)
    write_json(Path(output) / "summary.json", summary)
    write_json(Path(output) / "latency.json", {k: v["latency"] for k, v in summary["budgets"].items()})
    return summary


def aggregate_video_means(rows, budget):
    selected = [row for row in rows if row["tokens_per_tile_window"] == budget]
    keys = ["mse", "psnr_mean_db", "ms-ssim", "lpips", "lpips-vgg", "dists"]
    return {k: sum(r[k] for r in selected) / len(selected) for k in keys}
