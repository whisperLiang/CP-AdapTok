"""Consolidate actual byte-rate curves, common quality ranges and paired prefixes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import exp, log
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.v2.protocol import csv_at

TARGETS: dict[str, tuple[int, list[float]]] = {"global_psnr": (1, [25, 26, 28, 30]), "lpips": (-1, [0.45, 0.30, 0.28978]),
           "dists": (-1, [0.15, 0.20, 0.24])}
BASELINE = "Original AdapTok"


def read(path: Path):
    return json.loads(path.read_text())


def frontier(rows, metric):
    direction = TARGETS[metric][0]
    points: list[tuple[float, float]] = []
    best = float("-inf")
    for row in sorted(rows, key=lambda item: (item["actual_mbps"], -direction * item[metric])):
        rate, quality = row["actual_mbps"], direction * row[metric]
        if rate > 0 and quality > best and (not points or rate > points[-1][1]):
            points.append((quality, rate))
            best = quality
    return points


def at_quality(points, metric, target):
    value = target * TARGETS[metric][0]
    if not points or not points[0][0] <= value <= points[-1][0]:
        return None
    quality, rates = zip(*points, strict=True)
    return exp(float(np.interp(value, quality, [log(rate) for rate in rates])))


def common(left, right, metric):
    if not left or not right:
        return None
    lower, upper = max(left[0][0], right[0][0]), min(left[-1][0], right[-1][0])
    if lower >= upper:
        return None
    direction = TARGETS[metric][0]
    return {"signed": [lower, upper], "range": sorted((direction * lower, direction * upper))}


def bd_rate(left, right, span):
    if span is None or min(len(left), len(right)) < 3:
        return {"status": "OUT_OF_RANGE", "common_range": span["range"] if span else None}
    lower, upper = span["signed"]
    interpolators = []
    for curve in (left, right):
        quality, rates = zip(*curve, strict=True)
        interpolators.append(PchipInterpolator(quality, np.log(rates)))
    difference = interpolators[0].integrate(lower, upper) - interpolators[1].integrate(lower, upper)
    return {"status": "PASS", "common_range": span["range"],
            "rate_delta_percent": 100 * np.expm1(difference / (upper - lower)).item()}


def at_rate(rows, metric, rate):
    values = sorted((r["actual_mbps"], r[metric]) for r in rows)
    if not values or not values[0][0] <= rate <= values[-1][0]:
        return None
    rates, quality = zip(*values, strict=True)
    return float(np.interp(log(rate), np.log(rates), quality))


def archive_target(metric, target, original):
    # The named Original LPIPS target was rounded from its measured endpoint.
    # Use the stored endpoint across all curves and keep the reported name.
    if metric == "lpips" and target == 0.28978 and original:
        endpoint = max(row["lpips"] for row in original)
        if round(endpoint, 5) == target:
            return endpoint
    return target


def compare(curves):
    equal, bd, penalties = [], [], []
    original = curves[BASELINE]
    for name, rows in curves.items():
        for metric, (_, targets) in TARGETS.items():
            points, reference = frontier(rows, metric), frontier(original, metric)
            span = common(points, reference, metric)
            for target in targets:
                evaluated_target = archive_target(metric, target, original)
                candidate = at_quality(points, metric, evaluated_target)
                original_rate = at_quality(reference, metric, evaluated_target)
                equal.append({"method": name, "metric": metric, "target": target, "evaluated_target": evaluated_target,
                              "candidate_actual_mbps": candidate, "original_actual_mbps": original_rate,
                              "status": "PASS" if candidate is not None and original_rate is not None
                              else "OUT_OF_RANGE", "common_range": span["range"] if span else None})
            if name != BASELINE:
                bd.append({"method": name, "reference": BASELINE, "metric": metric,
                           **bd_rate(points, reference, span)})
        if name != "Official UF-LD":
            for row in rows:
                if row.get("budget_mbps") not in (5, 10, 20, 40):
                    continue
                when = row["actual_mbps"]
                comparison = {metric: at_rate(curves["Official UF-LD"], metric, when) for metric in TARGETS}
                penalties.append({"method": name, "budget_mbps": row["budget_mbps"], "actual_mbps": when,
                                  **{metric: row[metric] - val if val is not None else None
                                     for metric, val in comparison.items()}})
    return equal, bd, penalties


def adjacency(path):
    rows = csv_at(path)
    groups: dict[tuple[str, int], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "PASS":
            groups[row["video"], int(row["index"])][float(row["budget_mbps"])] = row
    budgets = (0, 2, 3, 3.594, 4, 5, 10, 20, 40)
    pairs = []
    for low, high in zip(budgets, budgets[1:]):
        for (video, frame), group in groups.items():
            if low in group and high in group:
                delta = {m: float(group[high][m]) - float(group[low][m]) for m in ("psnr", "lpips", "dists")}
                pairs.append({"low": low, "high": high, "video": video, "frame": frame, **delta})
    return pairs


def violations(rows):
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["low"], row["high"], "all")].append(row)
        groups[(row["low"], row["high"], row["video"])].append(row)
    result = []
    for (low, high, video), pairs in groups.items():
        for metric, direction in (("psnr", 1), ("lpips", -1), ("dists", -1)):
            delta = np.array([row[metric] for row in pairs])
            degradation = (-direction * delta).clip(min=0)
            result.append({"low": low, "high": high, "video": video, "metric": metric,
                           "mean_delta": delta.mean().item(), "frame_pairs": len(delta),
                           "violations": int(np.count_nonzero(degradation > 1e-7)),
                           "worst_degradation": degradation.max().item()})
    return result


def consolidated(results):
    audit = read(results / "baseline/baseline_audit.json")["curve_sources"]
    source = audit["original_and_temporal"]
    curves = {BASELINE: source["Original_AdapTok"], "HLA-Tok-T": source["T0_RA_S2"],
              "Official UF-LD": [x for x in audit["uf_ld"]["scores"] if x["method"] == "B3_gop32"],
              "Official DCVC-RT": [x for x in audit["rt"]["rows"] if x.get("schedule") == "gop32"],
              "UF-Progressive D0": audit["uf_d0"]["rows"], **audit["cc"]}
    curves = {name: values for name, values in curves.items() if values}
    fresh = {
        "Anchor-Fast": "anchor_fast/validation",
        "D1 / 500": "d1/seed1-step500-validation",
        "D1 / 2000": "d1/seed1-step2000-validation",
        "D1 / static": "d1/seed1-step2000-static",
        "D1 / perceptual": "d1/seed1-step2000-perceptual",
        "D1 / fixed": "d1/seed1-step2000-fixed",
        "D1 / fixed static": "d1/seed1-step2000-fixed-static",
        "D1 / Seed 2": "d1/seed2-step2000-static",
        "D1 / joint 4000": "d1/joint-seed1-step4000-validation",
        "D1 / L0 4000": "d1/l0-seed1-step4000-validation",
        "D1 / joint 6000": "d1/joint-seed1-step6000-validation",
        "D1 / joint 8000": "d1/joint-seed1-step8000-validation",
        "D1 / joint static": "d1/joint-seed1-final-static",
        "D1 / formal static": "d1/formal-seed1-final-static",
        "D1 / formal static fast": "d1/formal-seed1-final-static-fast",
        "D1 / formal E0": "d1/formal-seed1-final-e0",
        "D1 / formal E0 fast": "d1/formal-seed1-final-e0-fast",
        "D1 / Seed 2 final E0": "d1/formal-seed2-final-e0",
    }
    for path in sorted((results / "d1").glob("formal-seed*-step*-validation")):
        stage = path.name.split("-")
        fresh[f"D1 / {stage[1]} {stage[2][4:]}"] = str(path.relative_to(results))
    for step in (4000, 6000, 8000):
        fresh[f"D1 / seed2 {step}"] = f"d1/joint-seed2-step{step}-validation"
    measured = {}
    for name, location in fresh.items():
        path = results / location / "summary.json"
        if path.is_file():
            record = read(path)
            curves[name] = [row for row in record["quality"] if row["status"] == "PASS"]
            measured[name] = {"summary": location + "/summary.json", "quality": record["quality"],
                              "timing": record.get("timing"), "checkpoint_sha256": record["checkpoint_sha256"],
                              "model_tag": record.get("model_tag")}
            pairs = adjacency(path.with_name("quality.csv"))
            if pairs:
                write_csv(results / location / "adjacent_frames.csv", pairs)
                write_csv(results / location / "adjacency.csv", violations(pairs))
    return curves, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    curves, measured = consolidated(args.results)
    equal, bd, penalty = compare(curves)
    out = args.results / "evaluation"
    write_json(out / "rate_analysis.json", {"measured": measured, "curves": curves, "equal_quality": equal,
               "bd_rate": bd, "vs_official_ld": penalty,
               "scope": {"candidate": "hard frame budgets or complete Base; actual mean Mbps",
                         "official": "sequence-average measured fixed-QP rates",
                         "original": "actual F13W window average; 15-frame input buffer",
                         "equal_quality": "log-rate interpolation inside measured frontier; no extrapolation"}})
    write_csv(out / "equal_quality.csv", equal)
    write_csv(out / "bd_rate.csv", bd)
    write_csv(out / "official_ld_penalty.csv", penalty)


if __name__ == "__main__":
    main()
