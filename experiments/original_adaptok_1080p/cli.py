"""Run with python -m experiments.original_adaptok_1080p.cli --help."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml

from .artifacts import exclusive_writer, write_json
from .equivalence import check_equivalence
from .metrics import FullFrameMetrics
from .model_adapter import DEFAULT_REVISION, ROOT, OfficialAdapter, validate_budget
from .runner import evaluate_video, prepare_source, summarize_experiment
from .video_reader import file_sha256, manifest_sources, read_input, source_metadata


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["suite", "evaluate", "verify"])
    parser.add_argument("--config", default=str(ROOT / "configs/original_adaptok_1080p.yaml"))
    parser.add_argument("--checkpoint", help="unchanged official local model directory; otherwise pinned Hub snapshot")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output")
    parser.add_argument("--input", help="native 1080p MP4, image sequence, or raw YUV")
    parser.add_argument("--manifest")
    parser.add_argument("--content-ids", nargs="+")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--stride", type=int, choices=[96, 128])
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--tile-batch-size", type=int)
    parser.add_argument("--raw-width", type=int, default=1920)
    parser.add_argument("--raw-height", type=int, default=1080)
    parser.add_argument("--raw-pixel-format", default="yuv420p")
    parser.add_argument("--raw-color-range", choices=["limited", "full"])
    parser.add_argument("--raw-color-matrix", choices=["bt601", "bt709"])
    return parser.parse_args()


def configuration(args):
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ["checkpoint", "revision", "output", "manifest", "content_ids", "frames", "start", "budgets", "tile_batch_size"]:
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    if config["frames"] < 1 or config["start"] < 0 or config["tile_batch_size"] < 1 or config["warmup_batches"] < 1:
        raise ValueError("positive frame/batch/warmup counts and nonnegative start required")
    if config["dtype"] != "float32" or config["padding"] != "reflect":
        raise ValueError("this evaluator uses official FP32 weights and reflect padding")
    return config


def input_records(args, config):
    if not args.input:
        return manifest_sources(config["manifest"], config["content_ids"])
    path = Path(args.input).resolve()
    if path.suffix.lower() != ".yuv":
        metadata = source_metadata(path, args.fps)
        return [{**metadata, "content_id": f"single:{path.stem}"}]
    if args.raw_color_range is None or args.raw_color_matrix is None:
        raise ValueError("raw YUV requires --raw-color-range and --raw-color-matrix; never guess colorimetry")
    return [{"path": str(path), "content_id": f"single:{path.stem}", "sha256": file_sha256(path), "fps": args.fps,
             "raw_yuv": {"width": args.raw_width, "height": args.raw_height, "pixel_format": args.raw_pixel_format,
                         "color_range": args.raw_color_range, "color_matrix": args.raw_color_matrix}}]


def environment():
    packages = ["torch", "torchvision", "torchaudio", "numpy", "opencv-python-headless", "transformers", "huggingface-hub", "timm"]
    return {"python": platform.python_version(), "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
            "packages": {p: importlib.metadata.version(p) for p in packages}, "gpu": torch.cuda.get_device_name(),
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "tf32": torch.backends.cuda.matmul.allow_tf32, "autocast": False,
            "measurement_timezone": "Asia/Shanghai", "source_date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def warmup(adapter, source, config):
    from .video_reader import temporal_windows

    count = min(source.get("source_frames", adapter.frame_num), adapter.frame_num)
    clip, _ = read_input(source["path"], count, fps=source["fps"], raw_yuv=source.get("raw_yuv"), metadata=source)
    clip = next(temporal_windows(clip, adapter.frame_num)).frames
    # Real tiles and every budget; warmup is explicitly excluded from all benchmark windows.
    for _ in range(config["warmup_batches"]):
        tiles = clip[:, :, :adapter.input_size, :adapter.input_size].permute(1, 0, 2, 3)[None]
        tiles = tiles.expand(config["tile_batch_size"], -1, -1, -1, -1).contiguous().to(adapter.device)
        encoded = adapter.encode(tiles)
        for budget in config["budgets"]:
            adapter.decode(encoded, budget)
    torch.cuda.synchronize(adapter.device)


def execute(adapter, evaluator, sources, output, stride, budgets, frames, start, config):
    videos = [evaluate_video(s, adapter, evaluator, output, stride, budgets, frames, start, config) for s in sources]
    return summarize_experiment(output, videos, budgets)


def run_suite(adapter, evaluator, sources, output, config):
    # E1/E2 are independent single-budget timing measurements on the same first real 16-frame window.
    summaries = {}
    for name, stride in [("no_overlap", adapter.input_size), ("overlap", 96)]:
        summaries[name] = execute(adapter, evaluator, sources[:1], output / name, stride, [adapter.total_tokens], adapter.frame_num, config["start"], config)
    # E3/E4: all three videos, all native windows, all budgets; each tile encoded once per layout.
    for name, stride in [("final_1080p", adapter.input_size), ("token_ablation/overlap", 96)]:
        summaries[name] = execute(adapter, evaluator, sources, output / name, stride, config["budgets"], config["frames"], config["start"], config)
    return summaries


def main():
    args = arguments()
    config = configuration(args)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A process-level exclusive owner prevents competing writers while allowing crash recovery.
    with exclusive_writer(output):
        run(args, config, output)


def run(args, config, output):
    if not torch.cuda.is_available():
        write_json(output / "blocked.json", {"reason": "CUDA GPU unavailable", "measurements_completed": False})
        raise RuntimeError("CUDA GPU required for real official experiments; functional tests remain CPU executable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [prepare_source(s) for s in input_records(args, config)]
    if args.input and args.frames is None and "source_frames" in sources[0]:
        config["frames"] = sources[0]["source_frames"] - config["start"]
    adapter = OfficialAdapter(config.get("checkpoint"), config.get("revision", DEFAULT_REVISION), config["device"])
    for budget in config["budgets"]:
        validate_budget(budget, adapter.groups, adapter.tokens_per_group)
    # Actual checkpoint dimensions are authoritative; never override its configuration.
    write_json(output / "environment.json", environment())
    write_json(output / "checkpoint_manifest.json", adapter.manifest)
    write_json(output / "input_manifest.json", {"sources": sources, "requested_frames": config["frames"], "start": config["start"]})
    write_json(output / "resolved_config.json", config)
    equivalence = check_equivalence(adapter, sources[0])
    write_json(output / "smoke/official_equivalence.json", equivalence)
    if args.command != "verify":
        run_evaluation(args, config, output, adapter, sources)
    write_json(output / "checkpoint_manifest.json", adapter.verify_unchanged())
    print(json.dumps({"output": str(output), "official_equivalence": equivalence["passed"], "weights_unchanged": True}), flush=True)


def run_evaluation(args, config, output, adapter, sources):
    evaluator = FullFrameMetrics(adapter.device)
    write_json(output / "metric_manifest.json", evaluator.manifest())
    warmup(adapter, sources[0], config)
    if args.command == "suite":
        summaries = run_suite(adapter, evaluator, sources, output, config)
    else:
        summaries = {"evaluation": execute(adapter, evaluator, sources, output / "evaluation", args.stride or adapter.input_size,
                                             config["budgets"], config["frames"], config["start"], config)}
    write_json(output / "suite_summary.json", summaries)


if __name__ == "__main__":
    main()
