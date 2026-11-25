"""Derive matched-rate, matched-quality and adjacency evidence from measured frames."""

import argparse
from collections import defaultdict
from math import exp, log
from pathlib import Path
from statistics import mean

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.uf.inputs import BASELINE, OUTPUT
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

METRICS = {"global_psnr": 1, "lpips": -1, "dists": -1}
ADJACENT = (5, 10, 20, 40)


def interpolate(points: list[tuple[float, float]], target: float, log_x: bool = False) -> float | None:
    curve = sorted(points)
    if not curve or not curve[0][0] <= target <= curve[-1][0]:
        return None
    for value, estimate in curve:
        if value == target:
            return estimate
    coordinate = log if log_x else float
    for (lower, start), (upper, end) in zip(curve, curve[1:]):
        if lower < target < upper:
            alpha = (coordinate(target) - coordinate(lower)) / (coordinate(upper) - coordinate(lower))
            return start + alpha * (end - start)
    raise ValueError("interpolation interval not found")


def worsening(low: list[float], high: list[float], higher: bool) -> dict:
    direction = 1 if higher else -1
    drops = [(before - after) * direction for before, after in zip(low, high, strict=True)]
    count = sum(drop > 0 for drop in drops)
    return {"violations": count, "fraction": count / len(drops), "max_drop": max(0.0, max(drops))}


def _on_frontier(curve: list[dict], metric: str, direction: int) -> list[tuple[float, float]]:
    observations = sorted((row["actual_mbps"], -direction * row[metric]) for row in curve)
    best = float("-inf")
    frontier = []
    for rate, negative in observations:
        value = -negative
        if value > best and rate > 0:
            frontier.append((value, log(rate)))
            best = value
    return frontier


def compare(our: list[dict], reference: list[dict]) -> list[dict]:
    result = []
    for metric, direction in METRICS.items():
        fixed_rate = [(row["actual_mbps"], row[metric]) for row in reference]
        fixed_metric = _on_frontier(reference, metric, direction)
        for point in our:
            rate, quality = point["actual_mbps"], point[metric]
            ref_quality = interpolate(fixed_rate, rate, log_x=True)
            ref_log_rate = interpolate(fixed_metric, direction * quality)
            ref_rate = None if ref_log_rate is None else exp(ref_log_rate)
            result.append({"budget_mbps": point["budget_mbps"], "metric": metric, "candidate_actual_mbps": rate,
                           "candidate_quality": quality, "same_rate_ref_quality": ref_quality,
                           "same_rate_status": "OUT_OF_RANGE" if ref_quality is None else "PASS",
                           "same_rate_quality_gain": None if ref_quality is None else direction * (quality - ref_quality),
                           "equal_quality_ref_mbps": ref_rate,
                           "equal_quality_status": "OUT_OF_RANGE" if ref_rate is None else "PASS",
                           "equal_quality_rate_saving": None if ref_rate is None else (ref_rate - rate) / ref_rate})
    return result


def reference_qualities(reference: list[dict], candidate: list[dict]) -> list[dict]:
    result = []
    for metric, direction in METRICS.items():
        frontier = _on_frontier(candidate, metric, direction)
        for row in reference:
            quality, rate = row[metric], row["actual_mbps"]
            logarithm = interpolate(frontier, direction * quality)
            actual = None if logarithm is None else exp(logarithm)
            result.append({"metric": metric, "target_quality": quality, "reference_mbps": rate,
                           "candidate_mbps": actual, "status": "OUT_OF_RANGE" if actual is None else "PASS",
                           "candidate_rate_saving": None if actual is None else (rate - actual) / rate})
    return result


def _digest_matches(path: Path, hashes: dict) -> str:
    actual = sha256_file(path)
    expected = hashes[str(path.relative_to(ROOT))]
    if actual != expected:
        raise ValueError(f"protected archive changed: {path}")
    return actual


def archive_curves() -> tuple[dict, dict]:
    compact = ROOT / "results/hla_tok/compact"
    temporal = ROOT / "results/hla_tok/temporal"
    snapshots = json_at(OUTPUT / "evidence/historical_sha256.json")["tracked_sha256"]
    original_path, temporal_path = compact / "original_rd.csv", temporal / "bandwidth_comparison.csv"
    hashes = {"Original_AdapTok": _digest_matches(original_path, snapshots),
              "T0_RA_S2": _digest_matches(temporal_path, snapshots)}
    original = csv_at(original_path)
    temporal_rows = [row for row in csv_at(temporal_path) if row["method"] == "T0-RA S2"
                     and float(row["bandwidth"]) in ADJACENT]
    curves = {"Original_AdapTok": original, "T0_RA_S2": temporal_rows}
    for rows in curves.values():
        for row in rows:
            actual = int(row["total_bytes"]) * 240 / 174 / 1e6
            if int(row["frames"]) != 174 or abs(float(row["mbps_30fps"]) - actual) > 1e-9:
                raise ValueError("archived rate does not agree with transmitted application bytes")
    normalized = {name: [{key: float(row[key]) for key in METRICS} | {"actual_mbps": float(row["mbps_30fps"])}
                         for row in rows] for name, rows in curves.items()}
    return normalized, {"historical_commit": BASELINE, "mode": "validated read-only archived byte ledgers",
                        "fresh_rerun": False, "sha256": hashes, "Original_AdapTok_buffer_frames": 15,
                        "Original_AdapTok_buffer_ms_at_30fps": 500}


def adjacency(frames: list[dict]) -> dict:
    by_video: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for row in frames:
        if int(row["budget_mbps"]) in ADJACENT:
            by_video[row["video"]][(int(row["frame"]), int(row["budget_mbps"]))] = row
    result: dict[str, dict] = {}
    for video, entries in by_video.items():
        if any(row["status"] != "PASS" for row in entries.values()):
            result[video] = {"status": "FAIL"}
            continue
        indexes = sorted({frame for frame, _ in entries})
        result[video] = {}
        for low, high in zip(ADJACENT, ADJACENT[1:]):
            pair = {}
            for key in ("psnr", "lpips", "dists"):
                lower, upper = ([float(entries[index, budget][key]) for index in indexes] for budget in (low, high))
                pair[key] = worsening(lower, upper, higher=key == "psnr") | {"mean_delta": mean(upper) - mean(lower)}
            result[video][f"{low}_to_{high}"] = pair
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--official", type=Path, default=OUTPUT / "baseline/isolated/validation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archived, identity = archive_curves()
    for row in json_at(args.official)["scores"]:
        archived.setdefault(row["method"], []).append(row)
    measured = json_at(args.candidate / "summary.json")
    admitted = [row for row in measured["rows"] if row["status"] == "PASS"]
    enhanced = [row for row in admitted if row["budget_mbps"]]
    write_json(args.output, {"comparisons": {name: compare(enhanced, curve) for name, curve in archived.items()},
                            "at_reference_qualities": {name: reference_qualities(curve, admitted)
                                                       for name, curve in archived.items()},
                            "official_ld_at_reference_qualities": {
                                name: reference_qualities(archived[name], archived["B3_gop32"])
                                for name in ("Original_AdapTok", "T0_RA_S2", "B4")},
                            "adjacent_prefixes": adjacency(csv_at(args.candidate / "per_frame.csv")),
                            "comparison_scope": identity, "archive_curves": archived,
                            "rate_method": "interpolate only in observed range, linear in log actual Mbps",
                            "equal_quality_method": "rate-quality frontier, no extrapolation",
                            "read_counts": measured["read_counts"]})


if __name__ == "__main__":
    main()
