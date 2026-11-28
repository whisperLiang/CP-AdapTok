"""Independent single-budget profiling on real clips, without resident metric networks."""

import argparse
from pathlib import Path

import torch
import yaml

from common.artifacts import write_json

from .artifacts import exclusive_writer, write_samples, write_video
from .cli import warmup
from .model_adapter import ROOT, OfficialAdapter
from .profiling import latency_summary, measured
from .reconstruction import reconstruct_window
from .runner import budget_timing, experiment_identity, prepare_source
from .tiling import TileLayout
from .video_reader import manifest_sources, read_input


def benchmark_window(adapter, source, stride, directory, config):
    clip, metadata = read_input(source["path"], adapter.frame_num, config["start"], source["fps"], metadata=source)
    layout = TileLayout(1080, 1920, adapter.input_size, stride)
    torch.cuda.reset_peak_memory_stats(adapter.device)
    reconstructed, timing = reconstruct_window(clip, adapter, layout, [adapter.total_tokens], config["tile_batch_size"])
    peak = {"allocated_mib": torch.cuda.max_memory_allocated(adapter.device) / 2**20,
            "reserved_mib": torch.cuda.max_memory_reserved(adapter.device) / 2**20}

    def save():
        write_video(directory / "original.mp4", clip, source["fps"])
        write_video(directory / "reconstructed.mp4", reconstructed[adapter.total_tokens], source["fps"])
        write_samples(directory / "reconstruction_samples", clip, reconstructed[adapter.total_tokens])

    _, write_cost = measured(save, adapter.device)
    row = budget_timing(timing, adapter.total_tokens, adapter.frame_num, metadata["read_ms"], write_cost["wall_ms"], 0.)
    return {"input": metadata, "timing": row, "peak_memory": peak, "metric_networks_resident": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/adaptok-replay")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines/adaptok.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    with exclusive_writer(args.results):
        run(args.results, config)


def run(results, config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = OfficialAdapter(config["checkpoint"], config["revision"], config["device"])
    sources = [prepare_source(r) for r in manifest_sources(config["manifest"], config["content_ids"])]
    warmup(adapter, sources[0], {**config, "budgets": [adapter.total_tokens]})
    root = results / "benchmarks"
    root.mkdir(exist_ok=True)
    result = {}
    for stride in [adapter.input_size, 96]:
        windows = []
        for source in sources:
            directory = root / f"stride{stride}" / source["content_id"].replace(":", "_")
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "completed.json"
            identity = experiment_identity(source, adapter, stride, [adapter.total_tokens], adapter.frame_num, config["start"], config)
            if marker.exists():
                import json

                item = json.loads(marker.read_text())
                if item["identity"] != identity:
                    raise ValueError("refusing to reuse benchmark from a different configuration/input")
            else:
                print(f"benchmark {source['content_id']} stride={stride} K={adapter.total_tokens}", flush=True)
                item = benchmark_window(adapter, source, stride, directory, config)
                item["identity"] = identity
                write_json(marker, item)
            windows.append(item)
        result[str(stride)] = {"latency": latency_summary([w["timing"] for w in windows]),
                               "peak_allocated_mib": max(w["peak_memory"]["allocated_mib"] for w in windows),
                               "peak_reserved_mib": max(w["peak_memory"]["reserved_mib"] for w in windows),
                               "metric_networks_resident": False, "windows": windows,
                               "token_budget": adapter.total_tokens, "dtype": str(adapter.model.dtype),
                               "batch_size": config["tile_batch_size"], "sample_unit": "one native temporal window per video"}
    write_json(root / "summary.json", result)
    write_json(root / "checkpoint_manifest.json", adapter.verify_unchanged())
    print(root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
