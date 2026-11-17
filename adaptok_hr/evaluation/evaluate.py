"""Native RGB full-frame quality from real, independently decoded binary packets."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cp_adaptok.metrics import MetricSuite, summarize_latency

from ..codec.bitstream import HEADER, parse
from ..codec.codec import Codec
from ..data.dataset import read_clip
from ..data.manifest import load_index
from ..runtime import environment, file_hash, seed_everything, stable_hash, write_json
from ..training.checkpoint import load_model


def psnr(mse):
    return -10 * math.log10(max(mse, 1e-20))


def aggregate(rows):
    if not rows:
        return {"frames": 0}
    mse = sum(r["mse"] for r in rows) / len(rows)
    pixels = sum(r["pixels"] for r in rows)
    result = {"frames": len(rows), "mse": mse, "psnr_global": psnr(mse),
              "psnr_frame_mean": sum(r["psnr"] for r in rows) / len(rows),
              "bytes": sum(r["bytes"] for r in rows), "header_bytes": sum(r["header_bytes"] for r in rows),
              "payload_bytes": sum(r["payload_bytes"] for r in rows), "side_information_bytes": 0,
              "actual_bpp": sum(r["bytes"] for r in rows) * 8 / pixels,
              "theoretical_code_bpp": sum(r["tokens"] for r in rows) * 13 / pixels,
              "entropy_bpp": None, "real_bitstream": True}
    for name in ("lpips", "dists"):
        if all(r.get(name) is not None for r in rows):
            result[name] = sum(r[name] for r in rows) / len(rows)
    if all("encode_ms" in r for r in rows):
        result["sender_latency"] = summarize_latency([r["encode_ms"] for r in rows])
        result["receiver_latency"] = summarize_latency([r["decode_ms"] for r in rows])
        result["sender_plus_receiver_latency"] = summarize_latency([r["encode_ms"] + r["decode_ms"] for r in rows])
    return result


def group_rows(rows):
    selectors = {"I": lambda r: r["frame_type"] == "I", "P": lambda r: r["frame_type"] == "P",
                 "early_P": lambda r: 1 <= r["gop_position"] <= 7,
                 "late_P": lambda r: 24 <= r["gop_position"] <= 31,
                 "motion_low": lambda r: r["motion"] < .01,
                 "motion_medium": lambda r: .01 <= r["motion"] < .03,
                 "motion_high": lambda r: r["motion"] >= .03}
    return {key: aggregate([r for r in rows if predicate(r)]) for key, predicate in selectors.items()}


def histogram_summary(counts):
    values = np.asarray(counts, dtype=np.float64)
    probabilities = values[values > 0] / max(values.sum(), 1)
    return {"used_codes": int((values > 0).sum()), "codebook_usage": float((values > 0).mean()),
            "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
            "transmitted_tokens": int(values.sum()), "histogram": values.astype(int).tolist()}


def metric_manifest():
    import DISTS_pytorch
    import lpips

    paths = [Path(DISTS_pytorch.__file__).with_name("weights.pt"),
             Path(lpips.__file__).parent / "weights/v0.1/alex.pth"]
    paths += [Path(torch.hub.get_dir()) / "checkpoints" / p
              for p in ("alexnet-owt-7be5be79.pth", "vgg16-397923af.pth")]
    return {"versions": {p: importlib.metadata.version(p) for p in ("lpips", "DISTS-pytorch")},
            "weights": {str(p): file_hash(p) for p in paths}, "rgb_range": [0, 1],
            "lpips": "Alex v0.1, input RGB*2-1", "dists": "CP MetricSuite distributed alpha/beta; RGB [0,1]",
            "resolution": [1080, 1920], "metric_dtype": "float32", "metric_batch_size": 1}


def save_image(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (rgb.detach().float().cpu().squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(values).save(path)


def timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000


@torch.inference_mode()
def evaluate_source(model, source, budget, frames, metrics=None, amp_bf16=True, output=None):
    count = min(frames, source["original_frames"])
    clip, metadata = read_clip(source, 0, count, model.config.height, model.config.width)
    codec = Codec(model.eval(), amp_bf16)
    sending, receiving, rows = None, None, []
    histogram = np.zeros(model.config.codebook_size, dtype=np.int64)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, cpu_frame in enumerate(clip):
        frame = cpu_frame[None].to(device)
        (packet, sending), encode_ms = timed(lambda: codec.encode_frame(frame, sending, budget,
                                                                        source["content_id"]), device)
        (rgb, receiving), decode_ms = timed(lambda: codec.decode_frame(packet, receiving), device)
        if not torch.equal(sending.rgb, rgb) or not torch.equal(sending.feature, receiving.feature):
            raise RuntimeError("independent sender/receiver reconstructions diverged")
        decoded = rgb.float()
        if metrics is None:
            mse = (frame - decoded).square().mean().item()
            quality = {"mse": mse, "psnr": psnr(mse)}
        else:
            if tuple(frame.shape[-2:]) != (1080, 1920):
                raise ValueError("formal metrics require full 1080p")
            quality = metrics(frame, decoded)
        ids = parse(packet).ids
        histogram += np.bincount(ids, minlength=len(histogram))
        motion = float((cpu_frame - clip[index - 1]).abs().mean()) if index else 0.0
        row = {**quality, "content_id": source["content_id"], "frame": index,
               "frame_type": receiving.frame_type, "gop_position": receiving.gop_position,
               "tokens": budget, "bytes": len(packet), "header_bytes": HEADER.size,
               "payload_bytes": len(packet) - HEADER.size, "pixels": model.config.height * model.config.width,
               "encode_ms": encode_ms, "decode_ms": decode_ms, "motion": motion,
               "input_rgb_sha256": hashlib.sha256(cpu_frame.numpy().tobytes()).hexdigest(),
               "packet_sha256": hashlib.sha256(packet).hexdigest(), "state_chain": receiving.chain.hex()}
        rows.append(row)
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"frame_{index:04d}.bin").write_bytes(packet)
            if index in (0, 31, 63):
                save_image(output / f"original_{index:04d}.png", frame)
                save_image(output / f"reconstruction_{index:04d}.png", decoded)
    summary = {**aggregate(rows), **histogram_summary(histogram), "groups": group_rows(rows),
               "peak_gpu_mib_including_metrics": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
               "input": metadata, "at_least_64_contiguous_frames": count >= 64}
    if output is not None:
        (output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(output / "summary.json", summary)
    return summary, rows


@torch.inference_mode()
def validate_fast(model, index, ids, frames=8, amp_bf16=True):
    selected = [r for r in index["splits"]["validation"] if r["content_id"] in ids]
    if len(selected) != len(ids):
        raise ValueError("missing fixed validation source")
    budgets = {}
    for budget in model.config.budgets:
        rows = []
        for source in selected:
            _, measured = evaluate_source(model, source, budget, frames, amp_bf16=amp_bf16)
            rows.extend(measured)
        budgets[str(budget)] = aggregate(rows)
    return {"budgets": budgets, "metric_scope": "short validation RGB MSE; actual VQ packets; not final 64-frame evaluation"}


def seal_final_run(path, protocol):
    path = Path(path) / "final_test_seal.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("final test was sealed with a different checkpoint/protocol")
    write_json(path, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--split", choices=("train", "validation", "test", "historical"), default="validation")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--budgets", nargs="*", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seal-run", help="required for final test; permanently freezes this run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_everything(20261117)
    index = load_index(args.index)
    model, checkpoint = load_model(args.checkpoint, args.device)
    selected = [r for r in index["splits"][args.split] if not args.ids or r["content_id"] in args.ids]
    if not selected or (args.ids and len(selected) != len(args.ids)):
        raise ValueError("requested evaluation sources missing")
    budgets = args.budgets or model.config.budgets
    protocol = {"checkpoint_sha256": file_hash(args.checkpoint), "manifest_identity": index["identity"],
                "split": args.split, "ids": [r["content_id"] for r in selected], "frames": args.frames,
                "budgets": list(budgets), "amp_bf16": True, "metric_scope": "native full RGB, LPIPS-Alex, DISTS"}
    if args.split == "test":
        if not args.seal_run:
            raise ValueError("final test requires --seal-run after checkpoint selection")
        seal_final_run(args.seal_run, protocol)
    output = Path(args.output)
    identity = stable_hash(protocol)
    if (output / "run.json").exists():
        if json.loads((output / "run.json").read_text())["identity"] != identity:
            raise ValueError("evaluation output belongs to a different protocol")
    write_json(output / "run.json", {"identity": identity, "protocol": protocol, "environment": environment(),
                                     "completed_training_steps": checkpoint["completed_steps"],
                                     "timing": "synchronized wall; sender includes local decode; receiver separate; excludes disk,metrics,H2D",
                                     "motion_bins": [0.01, 0.03]})
    metrics = MetricSuite(args.device, names=("psnr", "lpips", "dists"))
    if metrics.errors:
        raise RuntimeError(f"required metric unavailable: {metrics.errors}")
    write_json(output / "metric_manifest.json", metric_manifest())
    summaries = {}
    for budget in budgets:
        rows, histogram, sources = [], np.zeros(8192, dtype=np.int64), {}
        for source in selected:
            directory = output / f"k{budget}" / source["content_id"].replace(":", "_")
            if (directory / "summary.json").exists():
                summary = json.loads((directory / "summary.json").read_text())
                measured = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            else:
                summary, measured = evaluate_source(model, source, budget, args.frames, metrics, output=directory)
            rows.extend(measured)
            histogram += np.asarray(summary["histogram"])
            sources[source["content_id"]] = summary
            print(json.dumps({"budget": budget, "source": source["content_id"], "frames": summary["frames"],
                              "psnr_global": summary["psnr_global"], "lpips": summary["lpips"], "dists": summary["dists"]}), flush=True)
        summaries[str(budget)] = {**aggregate(rows), **histogram_summary(histogram),
                                  "groups": group_rows(rows), "sources": sources}
    write_json(output / "summary.json", {"budgets": summaries, "protocol": protocol, "identity": identity})


if __name__ == "__main__":
    main()
