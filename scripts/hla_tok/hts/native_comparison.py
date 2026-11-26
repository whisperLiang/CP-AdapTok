"""Compare original byte-rate Pareto fronts over their measured overlap."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.uf_v2.rate_analysis import TARGETS, at_quality, bd_rate, common, frontier

CONFIG = Path("configs/hla_tok/hts/protocol.json")


def compared_points(candidates: list[dict], reference: list[dict], metric: str, upper_mbps: float) -> list[dict]:
    direction = TARGETS[metric][0]
    ordered = frontier(reference, metric)
    candidate_front = {(quality * direction, rate) for quality, rate in frontier(candidates, metric)}
    matches = []
    for row in candidates:
        rate, quality = row["actual_mbps"], row[metric]
        if rate >= upper_mbps or (quality, rate) not in candidate_front:
            continue
        matched_rate = at_quality(ordered, metric, quality)
        if matched_rate is not None:
            matches.append({"metric": metric, "quality": quality, "candidate_mbps": rate,
                            "reference_mbps": matched_rate,
                            "rate_saving_percent": 100 * (1 - rate / matched_rate)})
    return matches


def strict_fit(rows: list[dict], mbps: float, fps: float) -> dict:
    bytes_per_frame = Decimal(str(mbps)) * 1_000_000 / (8 * Decimal(str(fps)))
    fits = [row for row in rows if row["total_bytes"] <= int(bytes_per_frame * row["valid_frames"])]
    return {
        "units": len(rows), "affordable_units": len(fits),
        "valid_frames": sum(row["valid_frames"] for row in rows),
        "affordable_valid_frames": sum(row["valid_frames"] for row in fits),
        "i_units_over_budget": sum(row["unit_type"] == "I" for row in rows)
        - sum(row["unit_type"] == "I" for row in fits),
    }


def at_targets(points: list[dict], budgets: list[float], fps: float) -> list[dict]:
    rows = []
    for target in budgets:
        choices = [entry for entry in points if entry["summary"]["actual_mbps"] <= target]
        if not choices:
            rows.append({"budget_mbps": target, "status": "NO_MEASURED_ENDPOINT"})
            continue
        best = min(choices, key=lambda entry: entry["summary"]["lpips"])
        rows.append({"budget_mbps": target, "status": "MEASURED", **best["summary"],
                     **strict_fit(best["units"], target, fps)})
    return rows


def compare_profiles(entries: list[dict], config: dict) -> tuple[dict, list[dict], list[dict]]:
    by_model: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        point = entry["summary"]
        by_model.setdefault((point["profile"], point["model"]), []).append(entry)
    profiles: dict = {}
    targets: list[dict] = []
    pareto: list[dict] = []
    for (profile, model), points in sorted(by_model.items()):
        targets.extend({"profile": profile, "model": model, **row}
                       for row in at_targets(points, config["rate_targets_mbps"], config["report_fps"]))
        for metric, (direction, _) in TARGETS.items():
            ordered = frontier([entry["summary"] for entry in points], metric)
            pareto.extend({"profile": profile, "model": model, "metric": metric,
                           "quality": quality * direction, "actual_mbps": rate} for quality, rate in ordered)
        if model == "hts":
            profiles[profile] = compare_hts(points, by_model.get((profile, "ld"), []), config["phase0"])
    return profiles, targets, pareto


def compare_hts(entries: list[dict], ld_entries: list[dict], gate: dict) -> dict:
    candidates = [entry["summary"] for entry in entries]
    ld = [entry["summary"] for entry in ld_entries]
    output: dict[str, object] = {"hts_min_mbps": min(row["actual_mbps"] for row in candidates)}
    acceptable = [row for row in candidates if row["global_psnr"] >= gate["report_acceptable_quality_psnr_db"]]
    output["hts_lowest_measured_at_25_db"] = min(acceptable, key=lambda row: row["actual_mbps"], default=None)
    output["hts_vs_ld_low_rate_matches"] = {
        metric: compared_points(candidates, ld, metric, gate["low_rate_region_max_mbps"])
        for metric in gate["comparison_metrics"]}
    for metric in TARGETS:
        left, right = frontier(candidates, metric), frontier(ld, metric)
        output[f"hts_vs_ld_{metric}_bd_rate"] = bd_rate(left, right, common(left, right, metric))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    points = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*/*/point.json"))]
    profiles, targets, pareto = compare_profiles(points, config)
    write_csv(args.directory / "target_endpoints.csv", targets)
    write_csv(args.directory / "native_rd.csv", pareto)
    write_json(args.directory / "native_comparison.json", {"options": profiles, "points": len(points)})


if __name__ == "__main__":
    main()
