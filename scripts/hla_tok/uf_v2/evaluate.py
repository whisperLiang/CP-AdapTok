"""Full-max timing, independent quality replay, and physical byte accounting."""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS, MAX_APPLICATION_BYTES, AnchorModel, RefinementCodec, Segmentation
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedCodec, UnifiedRefinement
from hla_tok.uf.official import NativeDecoder, OfficialAssets, sha256_file
from hla_tok.uf.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD, BaseBudgetExceeded
from hla_tok.uf_v2.decoder import SharedRefinement
from hla_tok.uf_v2.native_weights import NativeAdjustments
from hla_tok.uf_v2.records import BatchRecords
from hla_tok.uf_v2.runtime import AnchorFast, Unified
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.anchor_eval import AnchorEncoder
from scripts.hla_tok.uf.baseline import distribution
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf.unified_eval import UnifiedEncoder
from scripts.hla_tok.v2.protocol import setup

QPS = (0, 8)
WARMUP = 4
SCOPES = (
    "encode_ms", "decode_ms", "base_ready_ms", "enhancement_ms", "render_ms", "total_ms",
)


def load_codec(options, assets):
    method, checkpoint, precision = options.method, options.checkpoint, options.precision
    if method.startswith("anchor"):
        if checkpoint is None:
            raise ValueError("Anchor evaluation requires an A2 checkpoint")
        model = AnchorModel().cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        codec = RefinementCodec(model, precision)
        return AnchorFast(assets, codec, QPS) if method == "anchor-fast" else AnchorEncoder(assets, codec, QPS)
    model = UnifiedRefinement().cuda().eval()
    segmentation = Segmentation(8, 8, ranked=not options.fixed)
    if method == "d0":
        return UnifiedEncoder(assets, UnifiedCodec(model, segmentation), QPS)
    table = StaticAsset.load(options.entropy_table).tables if options.entropy_table else None
    if options.raw and table is None:
        raise ValueError("raw fallback requires the static-probability asset")
    weights = None
    if options.sensitivity:
        data = json.loads(options.sensitivity.read_text())
        weights = {True: np.asarray(data["I"]), False: np.asarray(data["P"])}
    bank = BatchRecords(tuple(model.scales().tolist()), segmentation, table, options.raw, weights) if (
        table or weights) else None
    if method != "d1":
        return Unified(assets, model, QPS, bank=bank, segmentation=segmentation)
    refiner = SharedRefinement().cuda().half().to(memory_format=torch.channels_last).eval()
    adjustments = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["quantizer"])
        refiner.load_state_dict(state["refiner"])
        if "native_adjustments" in state:
            adjustments = NativeAdjustments().cuda()
            adjustments.load_state_dict(state["native_adjustments"])
    return Unified(assets, model, QPS, refiner, bank=bank, segmentation=segmentation, adjustments=adjustments)


def decoder_for(codec, assets):
    return codec.new_decoder() if hasattr(codec, "new_decoder") else NativeDecoder(assets)


def allowance(budget: float) -> int:
    return int(Decimal(str(budget)) * 1_000_000 / 240)


def encode(codec, rgb, index, cap=MAX_APPLICATION_BYTES):
    intra, reset = schedule(index)
    if cap != MAX_APPLICATION_BYTES:
        return codec.encode(rgb, index, intra, reset, cap=cap)
    return codec.encode(rgb, index, intra, reset)


def receive(codec, decoder, family, cap):
    try:
        prefix, records = family.prefix(cap)
    except BaseBudgetExceeded:
        decoder.order.desync = True
        return None, {"status": "BASE_BUDGET_EXCEEDED", "base_bytes": family.base_end}
    start = perf_counter()
    receiver = codec.receiver(decoder)
    try:
        receiver.feed(prefix[:family.base_end])
        if receiver.decode_available() != 1:
            raise RuntimeError("expected exactly one base commit")
        base_ready = perf_counter()
        receiver.feed(prefix[family.base_end:])
        if receiver.decode_available() != records:
            raise RuntimeError("prefix boundary disagrees with record count")
        receiver.finish()
    except ValueError as error:
        if "DESYNC" not in str(error):
            raise
        return None, {"status": "DESYNC", "base_bytes": family.base_end, "bytes": len(prefix)}
    enhanced = perf_counter()
    image = receiver.render()
    torch.cuda.current_stream().synchronize()
    end = perf_counter()
    return (receiver, image), {
        "status": "PASS", "bytes": len(prefix), "base_bytes": family.base_end,
        "native_joint_y_z_bytes": family.base_end - APP_BASE_OVERHEAD,
        "base_header_bytes": APP_BASE_OVERHEAD, "enhancement_headers": records * RECORD_OVERHEAD,
        "enhancement_payload": len(prefix) - family.base_end - records * RECORD_OVERHEAD,
        "full_max_bytes": len(family.raw), "records": records,
        "integer_symbols": receiver.symbol_count, "unused_bytes": cap - len(prefix),
        "base_ready_ms": 1000 * (base_ready - start), "enhancement_ms": 1000 * (enhanced - base_ready),
        "render_ms": 1000 * (end - enhanced), "decode_ms": 1000 * (end - start),
    }


def annotate(data, clip, index, budget, cap):
    return {"video": clip.video, "index": index, "type": "I" if schedule(index)[0] else "P",
            "budget_mbps": budget, "cap": cap, **data}


def phase_summary(rows):
    out = {"frames": len(rows), "fps": 1000 / mean(row["total_ms"] for row in rows)}
    for key in SCOPES:
        out.update(distribution((row[key] for row in rows), key))
    for key in ("native_encode_ms", "analysis_copy_ms", "group_rank_pack_ms", "quant_rank_entropy_pack_ms"):
        if key in rows[0]:
            out.update(distribution((row[key] for row in rows), key))
    return out


def timing(codec, assets, reader, clips, budgets, repeats, frames_limit, demand):
    decoder = decoder_for(codec, assets)
    rows = []
    for rep in range(repeats):
        for clip in clips:
            frames = [source_frame(reader, clip, n) for n in range(min(clip.length, frames_limit))]
            for budget in budgets:
                cap = BUDGETS[budget]
                encode_cap = cap if demand else MAX_APPLICATION_BYTES
                codec.reset()
                decoder.reset()
                for index, rgb in enumerate(frames[:WARMUP]):
                    out = encode(codec, rgb, index, encode_cap)
                    receive(codec, decoder, out.family, cap)
                codec.reset()
                decoder.reset()
                for index, rgb in enumerate(frames):
                    torch.cuda.synchronize()
                    start = perf_counter()
                    out = encode(codec, rgb, index, encode_cap)
                    _, stats = receive(codec, decoder, out.family, cap)
                    total = (perf_counter() - start) * 1000
                    if stats["status"] != "PASS":
                        raise ValueError(f"timed prefix failed to close: {clip.video} {index}: {stats}")
                    rows.append(annotate(stats | out.phase_ms | {"total_ms": total, "repeat": rep + 1,
                                "rgb_ready_ms": total - stats["decode_ms"] + stats["base_ready_ms"]},
                                         clip, index, budget, cap))
        print("timing repetition", rep + 1, flush=True)
    return rows


def summarize_timing(rows, budgets, repeats):
    result = {}
    for budget in budgets:
        group = [row for row in rows if row["budget_mbps"] == budget]
        result[budget] = {
            "pooled": phase_summary(group),
            "repetitions": [phase_summary([row for row in group if row["repeat"] == n + 1])
                            for n in range(repeats)],
            "steady": phase_summary([row for row in group if row["index"] >= WARMUP]),
            "by_type": {kind: phase_summary([row for row in group if row["type"] == kind])
                        for kind in ("I", "P")},
            "first_base": distribution((row["rgb_ready_ms"] for row in group), "rgb_ready_ms"),
        }
    return result


def quality_replay(codec, assets, reader, clips, budgets, frames_limit):
    metric = NativeQuality()
    decoders = {budget: decoder_for(codec, assets) for budget in budgets}
    rows = []
    for clip in clips:
        codec.reset()
        for decoder in decoders.values():
            decoder.reset()
        for index in range(min(clip.length, frames_limit)):
            rgb = source_frame(reader, clip, index)
            metric.prepare(rgb)
            encoded = encode(codec, rgb, index)
            for budget, decoder in decoders.items():
                cap = allowance(budget) if budget else encoded.family.base_end
                render, stats = receive(codec, decoder, encoded.family, cap)
                if render is not None:
                    state, image = render
                    if not codec.matches(encoded, decoder, state):
                        raise ValueError("independent reference state differs from encoder")
                    stats |= metric.score(image)
                rows.append(annotate(stats, clip, index, budget, cap))
        print("quality", clip.video, min(clip.length, frames_limit), flush=True)
    return rows


def summarize_quality(rows, budgets):
    result = []
    for budget in budgets:
        group = [row for row in rows if row["budget_mbps"] == budget]
        successes = [row for row in group if row["status"] == "PASS"]
        stats = {"budget_mbps": budget, "planned": len(group), "decoded": len(successes),
                 "base_exceeded": sum(row["status"] == "BASE_BUDGET_EXCEEDED" for row in group),
                 "desync": sum(row["status"] == "DESYNC" for row in group)}
        if len(successes) == len(group):
            byte_sum = sum(row["bytes"] for row in group)
            stats |= summarize(group) | {"status": "PASS", "actual_mbps": byte_sum * 240 / len(group) / 1e6,
                                         "actual_bpp": byte_sum * 8 / (len(group) * 1920 * 1080),
                                         "total_bytes": byte_sum, "max": max(row["bytes"] for row in group),
                                         "unused_bytes": mean(row["unused_bytes"] for row in group)}
        else:
            stats["status"] = "BASE_BUDGET_EXCEEDED"
        result.append(stats)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("anchor", "anchor-fast", "d0", "d0-fast", "d1"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--split", choices=("validation", "calibration", "long"), default="validation")
    parser.add_argument("--frame-limit", type=int, default=10_000)
    parser.add_argument("--clip-limit", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timing-budgets", type=int, nargs="+", choices=list(BUDGETS), default=list(BUDGETS))
    parser.add_argument("--quality", type=float, nargs="+")
    parser.add_argument("--demand", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--entropy-table", type=Path)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    setup()
    reader, plans = inputs()
    clips = plans[options.split][:options.clip_limit]
    assets = OfficialAssets.verify(options.weights)
    codec = load_codec(options, assets)
    report = {"method": options.method, "precision": "native FP16; Anchor " + options.precision.upper()
              if options.method.startswith("anchor") else "native/display FP16; latent FP32",
              "qp": QPS, "split": options.split, "checkpoint_sha256": sha256_file(options.checkpoint)
              if options.checkpoint else None, "rate_fps": 30, "warmup": WARMUP,
              "static_cdf_file_sha256": sha256_file(options.entropy_table) if options.entropy_table else None,
              "raw_fallback": options.raw,
              "sensitivity_sha256": sha256_file(options.sensitivity) if options.sensitivity else None,
              "schedule": "fixed_scan" if options.fixed else "gain_per_byte", "manifest_hashes": reader.hashes,
              "size_hw": [1080, 1920], "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "mode": "on-demand enhancement" if options.demand else "full-max"}
    if options.repetitions:
        torch.cuda.reset_peak_memory_stats()
        rows = timing(codec, assets, reader, clips, options.timing_budgets, options.repetitions, options.frame_limit,
                      options.demand)
        write_csv(options.output / "timing.csv", rows)
        report["timing"] = summarize_timing(rows, options.timing_budgets, options.repetitions)
        report["timing_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if options.quality is not None:
        rows = quality_replay(codec, assets, reader, clips, options.quality, options.frame_limit)
        write_csv(options.output / "quality.csv", rows)
        report["quality"] = summarize_quality(rows, options.quality)
    write_json(options.output / "summary.json", report | {"read_counts": reader.counts})


if __name__ == "__main__":
    main()
