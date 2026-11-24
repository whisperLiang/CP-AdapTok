"""Native-only evaluation with separate dense, continuous and physical Fixed13 rows."""

from collections import defaultdict
from random import Random
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hla_tok import HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import psnr, summarize_quality
from hla_tok.evaluation.reconstruction import CodeUsage, save_native
from hla_tok.evaluation.representation_diagnostics import summarize_prefix
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger
from scripts.hla_tok.v3.protocol import NATIVE

REPRESENTATION_FIELDS = ("effective_rank", "stable_rank", "centered_variance", "common_energy_fraction")
MAIN = ("global_psnr", "lpips", "dists")


def _native_reconstructions(model, source, budgets, quantized, bypass):
    if not isinstance(model, HLATok):
        yield ("dense", None), model(source), {}, None, None
        return
    tokens = model.encode_training(source, max(budgets), quantize=quantized)
    receiver = model.receiver() if quantized else None
    for budget in budgets:
        continuous = tokens.continuous[:, :budget]
        if receiver is None:
            yield ("continuous", budget), model.decode_training(tokens, budget), {}, continuous, None
            continue
        ids = tokens.ids[:, :budget]
        wire = pack_tokens_13bit(ids[0].cpu().tolist())
        decoded = receiver.decode_bytes(wire, budget, NATIVE)
        rate = payload_ledger(len(wire), budget, 1, NATIVE.height, NATIVE.width)
        rate["quantization_mse"] = F.mse_loss(tokens.values[:, :budget].float(), continuous.float()).item()
        yield ("fixed13", budget), decoded, rate, continuous, ids
        if bypass:
            yield ("same_checkpoint_bypass", budget), model._decode(continuous, NATIVE), {}, continuous, None


def _bootstrap(rows, output):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video"]].append(row)
    groups = [summarize_quality(group) for group in grouped.values()]
    generator = Random(6309)
    draws = {field: [] for field in MAIN}
    for _ in range(1000):
        chosen = generator.choices(groups, k=len(groups))
        count = sum(group["frames"] for group in chosen)
        for field in MAIN:
            key = "mse" if field == "global_psnr" else field
            sample = sum(group[key] * group["frames"] for group in chosen) / count
            draws[field].append(psnr(sample) if key == "mse" else sample)
    write_json(output, {"unit": "video", "videos": list(grouped), "draws": 1000,
                        "percentile_95": {field: np.percentile(samples, (2.5, 97.5)).tolist()
                                          for field, samples in draws.items()}})


def _summaries(rows, usage, output, subset):
    summaries = []
    for (mode, budget), frames in rows.items():
        summary = {"mode": mode, "budget": budget, "matched_174": not subset} | summarize_quality(frames)
        for field in REPRESENTATION_FIELDS:
            if field in frames[0]:
                summary[field] = mean(row[field] for row in frames)
        if mode == "fixed13":
            stats = usage[budget].report()
            overall = stats["overall"]
            total = sum(row["total_bytes"] for row in frames)
            count = len(frames)
            summary |= payload_ledger(total, budget * count, count, NATIVE.height, NATIVE.width)
            summary |= {field: overall[field] for field in
                        ("active_ratio", "dead_ratio", "entropy_bits", "perplexity")}
            summary["top_code_fraction"] = overall["top_counts"][0] / overall["token_count"]
            summary["same_code_fraction"] = stats["same_code_fraction"]
            summary["quantization_mse"] = mean(row["quantization_mse"] for row in frames)
        if not subset:
            _bootstrap(frames, output / f"video_uncertainty_{mode}_{budget}.json")
        summaries.append(summary)
    return summaries


@torch.inference_mode()
def evaluate(model: nn.Module, reader, cohort, quality, output, *, budgets=(), quantized=False, subset=False,
             bypass=False):
    model.eval()
    rows: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    usage = {budget: CodeUsage(8192) for budget in budgets} if quantized else {}
    before = dict(reader.counts)
    try:
        for video, count in cohort.items():
            indices = (0, count // 2, count - 1) if subset else range(count)
            for index in indices:
                source = transfer(reader.read("validation", video, index)[None])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    images = list(_native_reconstructions(model, source, budgets, quantized, bypass))
                for (mode, budget), image, rate, continuous, ids in images:
                    row = {"mode": mode, "budget": budget, "video": video, "frame": index}
                    if ids is not None:
                        usage[budget].update(ids)
                    if continuous is not None and not subset and mode != "same_checkpoint_bypass":
                        stats = summarize_prefix(continuous, NATIVE.num_regions)
                        row |= {field: stats[field].mean().item() for field in REPRESENTATION_FIELDS}
                    rows[(mode, budget)].append(row | rate | quality(source, image))
                    if not subset and index == 32 and video.endswith("car-roundabout"):
                        save_native(output / "samples", f"{mode}_{budget or 'native'}", image)
                        save_native(output / "samples", "source", source)
        expected = 9 if subset else 174
        if any(len(frames) != expected for frames in rows.values()):
            raise ValueError("evaluation did not use exactly the frozen native frame count")
        report = _summaries(rows, usage, output, subset)
        write_csv(output / "per_frame.csv", [row for frames in rows.values() for row in frames])
        write_json(output / "metrics.json", report)
        return report
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
