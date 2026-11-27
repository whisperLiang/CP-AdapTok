"""Full-max timing, independent quality replay, and physical byte accounting."""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from baselines.dcvc_uf.frames import schedule
from baselines.dcvc_uf.official import OfficialAssets
from baselines.hla_tok_uf_v2.decoder import SharedRefinement
from baselines.hla_tok_uf_v2.latent import BUDGETS, MAX_APPLICATION_BYTES, UnifiedRefinement
from baselines.hla_tok_uf_v2.native_weights import NativeAdjustments
from baselines.hla_tok_uf_v2.runtime import Unified
from baselines.hla_tok_uf_v2.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD, BaseBudgetExceeded
from common.artifacts import write_csv, write_json
from common.inputs import inputs, json_at, setup, source_frame
from common.integrity import sha256_file
from common.metrics import NativeQuality, summarize
from common.paths import ROOT
from common.performance import distribution

D1_CONFIG = ROOT / "configs/baselines/d1.json"
D1_CHECKPOINT = ROOT / "checkpoints/hla_tok_uf_v2/d1-seed2.pt"
QPS = (0, 8)
WARMUP = 4
SCOPES = (
    "encode_ms", "decode_ms", "base_ready_ms", "enhancement_ms", "render_ms", "total_ms",
)


def load_codec(checkpoint: Path, assets: OfficialAssets) -> Unified:
    if sha256_file(checkpoint) != json_at(D1_CONFIG)["sha256"]:
        raise ValueError("D1 inference checkpoint differs from the frozen Seed 2 E0 manifest")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UnifiedRefinement().cuda().eval()
    refiner = SharedRefinement().cuda().half().eval()
    torch.nn.utils.convert_conv2d_weight_memory_format(refiner, torch.channels_last)
    adjustments = NativeAdjustments().cuda()
    model.load_state_dict(state["quantizer"])
    refiner.load_state_dict(state["refiner"])
    adjustments.load_state_dict(state["native_adjustments"])
    return Unified(assets, model, QPS, refiner, adjustments=adjustments)


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
    decoder = codec.new_decoder()
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
    decoders = {budget: codec.new_decoder() for budget in budgets}
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
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=D1_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "calibration", "long"), default="validation")
    parser.add_argument("--frame-limit", type=int, default=10_000)
    parser.add_argument("--clip-limit", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timing-budgets", type=int, nargs="+", choices=list(BUDGETS), default=list(BUDGETS))
    parser.add_argument("--quality", type=float, nargs="+")
    parser.add_argument("--demand", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    setup()
    reader, plans = inputs()
    clips = plans[options.split][:options.clip_limit]
    assets = OfficialAssets.verify(options.weights)
    codec = load_codec(options.checkpoint, assets)
    report = {"method": "d1", "precision": "native/display FP16; latent FP32",
              "qp": QPS, "split": options.split, "checkpoint_sha256": sha256_file(options.checkpoint)
              if options.checkpoint else None, "rate_fps": 30, "warmup": WARMUP,
              "model_tag": codec.tag.hex(),
              "schedule": "gain_per_byte", "manifest_hashes": reader.hashes,
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
