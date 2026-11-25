"""Run official single-frame FP16 codec on frozen Calibration/Validation."""

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.official import NativeDecoder, NativeEncoder, NativeNAL, OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import Clip, source_frame
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import csv_at, json_at, setup

QP_GRID = (0, 8, 16, 24, 32, 40, 48, 56, 63)
BASE_CAP = 9212
RATE_FPS = 30


@dataclass(frozen=True)
class Point:
    method: str
    qp_i: int
    qp_p: int
    intra_period: int
    reset_interval: int = 32

    @property
    def video(self) -> bool:
        return self.intra_period != 1


def distribution(values: Iterable[float], label: str) -> dict:
    array = np.asarray(list(values))
    return {f"{label}_mean": float(array.mean()), f"{label}_max": float(array.max())} | {
        f"{label}_p{pct}": float(np.percentile(array, pct)) for pct in (50, 95, 99)}


def run_clip(encoder: NativeEncoder, decoder: NativeDecoder, reader: AuditedFrames, clip: Clip,
             point: Point, quality: NativeQuality | None) -> list[dict]:
    writer, parser = NativeNAL(), NativeNAL()
    encoder.reset()
    decoder.reset()
    rows = []
    for index in range(clip.length):
        rgb = source_frame(reader, clip, index)
        torch.cuda.synchronize()
        intra, reset = schedule(index, point.intra_period, point.reset_interval)
        start = perf_counter()
        base = encoder.encode(rgb, index, point.qp_i if intra else point.qp_p, intra, reset)
        raw = writer.encode(base)
        encoded = perf_counter()
        decoded = decoder.decode(parser.decode(raw, index))
        end = perf_counter()
        row = {"video": clip.video, "source_index": clip.start + index, "frame": index,
               "type": "I" if intra else "P", "memory_reset": reset,
               "bytes": len(raw), "payload_bytes": len(base.payload), "nal_bytes": len(raw) - len(base.payload),
               "encode_ms": (encoded - start) * 1000, "decode_ms": (end - encoded) * 1000,
               "total_ms": (end - start) * 1000}
        if quality is not None:
            quality.prepare(rgb)
            row |= quality.score(decoded)
        rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    measured = len(rows)
    result = summarize(rows) if "mse" in rows[0] else {"frames": measured}
    total = sum(row["bytes"] for row in rows)
    return result | distribution((row["bytes"] for row in rows), "bytes") | {
        "payload_bytes_max": max(row["payload_bytes"] for row in rows),
        "actual_mbps": total * 8 * RATE_FPS / measured / 1e6,
        "actual_bpp": total * 8 / (measured * 1920 * 1080),
        "nal_bytes": sum(row["nal_bytes"] for row in rows),
        "encode_fps": 1000 / mean(row["encode_ms"] for row in rows),
        "decode_fps": 1000 / mean(row["decode_ms"] for row in rows),
        "serial_fps": 1000 / mean(row["total_ms"] for row in rows),
    } | distribution((row["total_ms"] for row in rows), "total_ms")


def evaluate(assets: OfficialAssets, reader: AuditedFrames, clips: list[Clip], point: Point,
             quality: NativeQuality | None) -> tuple[list[dict], dict]:
    torch.cuda.reset_peak_memory_stats()
    encoder, decoder = (cls(assets, point.video) for cls in (NativeEncoder, NativeDecoder))
    warmup = Clip(clips[0].split, clips[0].video, clips[0].start, 4)
    run_clip(encoder, decoder, reader, warmup, point, None)
    rows = [row for clip in clips for row in run_clip(encoder, decoder, reader, clip, point, quality)]
    scores = summary(rows) | vars(point) | {
        "precision": "FP16", "warmup_frames": 4, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "intra_parameters": encoder.parameters(intra=True),
        "video_parameters": encoder.parameters(intra=False) if point.video else None,
    }
    subsets = {label: [row for row in rows if row["type"] == label] for label in ("I", "P")}
    return [row | vars(point) for row in rows], scores | {
        "by_type": {label: summary(subset) for label, subset in subsets.items() if subset},
    }


def select_calibration(records: list[dict]) -> Point:
    def application_max(row: dict, kind: str) -> float:
        return row["by_type"][kind]["payload_bytes_max"] + APP_BASE_OVERHEAD

    eligible_i = [row for row in records if application_max(row, "I") <= BASE_CAP]
    intra = max(eligible_i, key=lambda row: row["qp_i"]) if eligible_i else min(
        records, key=lambda row: application_max(row, "I"))
    eligible_p = [row for row in records if application_max(row, "P") <= BASE_CAP and row["actual_mbps"] <= 2]
    inter = max(eligible_p, key=lambda row: row["qp_p"]) if eligible_p else min(
        records, key=lambda row: application_max(row, "P"))
    return Point("base_locked", intra["qp_i"], inter["qp_p"], 32)


def calibrate_pair(output: Path, assets: OfficialAssets, reader: AuditedFrames, clips: list[Clip],
                   scores: list[dict]) -> None:
    from dataclasses import replace

    first = select_calibration(scores)
    candidates = sorted({row["qp_p"] for row in scores if row["qp_p"] <= first.qp_p}, reverse=True)
    attempts = []
    for qp in candidates:
        _, score = evaluate(assets, reader, clips, replace(first, qp_p=qp), None)
        score["app_max"] = {kind: data["payload_bytes_max"] + APP_BASE_OVERHEAD
                            for kind, data in score["by_type"].items()}
        score["bound_pass"] = max(score["app_max"].values()) <= BASE_CAP
        attempts.append(score)
        if score["bound_pass"] and score["actual_mbps"] <= 2:
            break
    chosen = attempts[-1] if attempts[-1]["bound_pass"] else min(attempts, key=lambda row: max(row["app_max"].values()))
    write_json(output / "base_lock.json", {"selected_from": "Calibration (frozen Train holdout)",
               "cap_bytes": BASE_CAP, "application_overhead": APP_BASE_OVERHEAD,
               "bound_pass": chosen["bound_pass"], "score": chosen, "closed_loop_attempts": attempts,
               "online_retries": 0, "read_counts": reader.counts})


def run(mode: str, output: Path, assets: OfficialAssets, qps: tuple[int, ...]) -> None:
    setup()
    reader, plans = inputs()
    if mode == "lock":
        scores = json_at(output / "calibration.json")["scores"]
        rows = csv_at(output / "calibration_per_frame.csv")
        for score in scores:
            for kind, subset in score["by_type"].items():
                subset["payload_bytes_max"] = max(int(row["payload_bytes"]) for row in rows
                                                  if int(row["qp_i"]) == score["qp_i"] and row["type"] == kind)
        calibrate_pair(output, assets, reader, plans["calibration"], scores)
        return
    clips = plans[mode]
    quality = None if mode == "calibration" else NativeQuality()
    modes = (("calibration_grid", 32),) if quality is None else (("B3_gop32", 32), ("B3_default", 0), ("B4", 1))
    rows, scores = [], []
    for label, period in modes:
        for qp in qps:
            batch, result = evaluate(assets, reader, clips, Point(label, qp, qp, period), quality)
            rows.extend(batch)
            scores.append(result)
            print(label, qp, round(result["actual_mbps"], 4), round(result["serial_fps"], 1), flush=True)
    write_csv(output / f"{mode}_per_frame.csv", rows)
    write_json(output / f"{mode}.json", {"scores": scores, "read_counts": reader.counts, "manifests": reader.hashes})
    if mode == "calibration":
        calibrate_pair(output, assets, reader, clips, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibration", "validation", "lock"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT / "baseline")
    parser.add_argument("--qps", type=int, nargs="+", default=QP_GRID)
    args = parser.parse_args()
    run(args.mode, args.output, OfficialAssets.verify(checkpoint_directory(args.weights)), tuple(args.qps))


if __name__ == "__main__":
    main()
