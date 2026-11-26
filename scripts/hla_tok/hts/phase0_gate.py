"""Decide whether complete Calibration evidence admits native HT-S integration."""

import argparse
from collections.abc import Iterable
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.hts.native_comparison import CONFIG, compare_profiles
from scripts.hla_tok.v2.protocol import json_at


def point_key(row: dict) -> tuple[str, str, int, int]:
    return row["model"], row["profile"], row["qp_i"], row["qp_p"]


def grid(config: dict) -> set[tuple[str, str, int, int]]:
    first, last = config["native_qp_range"]
    profiles, mixed = config["native_profiles"], config["low_rate_mixed_qps"]
    direct = {(variant, profile, qp, qp) for variant in config["models"]
              for profile in profiles for qp in range(first, last + 1)}
    return direct | {("hts", profile, i, p) for profile in profiles for i in mixed for p in mixed}


def fully_measured_control(control: dict, frames: int) -> dict:
    matches = [row for row in control["quality"] if row["budget_mbps"] == 0 and row["decoded"] == frames]
    if len(matches) != 1:
        raise ValueError("missing complete Calibration Base-only control")
    return matches[0]


def old_base_improvements(candidates: Iterable[dict], old: dict, gate: dict) -> list[dict]:
    improvements = []
    for point in candidates:
        rate = point["actual_mbps"]
        if rate >= min(old["actual_mbps"], gate["low_rate_region_max_mbps"]):
            continue
        gains = {"global_psnr": point["global_psnr"] - old["global_psnr"], "lpips": old["lpips"] - point["lpips"]}
        if max(gains.values()) >= 0:
            improvements.append({
                "model": point["model"], "profile": point["profile"], "qp_i": point["qp_i"], "qp_p": point["qp_p"],
                "rate_mbps": rate, "control_mbps": old["actual_mbps"],
                "rate_saving_percent": 100 * (1 - rate / old["actual_mbps"]), "quality_gains": gains,
            })
    return improvements


def select_qp(rows: list[dict], allowed_budgets: list[float]) -> list[dict]:
    configurations = []
    for budget in allowed_budgets:
        legal = [row for row in rows if row["actual_mbps"] <= budget]
        if not legal:
            configurations.append({"budget_mbps": budget, "status": "UNAVAILABLE"})
            continue
        best = min(legal, key=lambda row: (row["lpips"], row["actual_mbps"], row["qp_i"], row["qp_p"], row["profile"]))
        configurations.append({"budget_mbps": budget, "status": "MEASURED", "calibration": best})
    return configurations


def inspect(directory: Path, config: dict) -> dict:
    points = [json_at(path) for path in sorted(directory.glob("*/*/point.json"))]
    rows = [point["summary"] for point in points]
    missing = sorted(grid(config) - {point_key(row) for row in rows})
    evidence = ("native_correctness.json", "native_continuous_capacity.csv", "legacy/original/original_rd.csv",
                "legacy/a2/summary.json", "legacy/a4/summary.json")
    absent = [name for name in evidence if not (directory / name).is_file()]
    if missing or absent:
        return {"verdict": "BLOCKED", "missing_native_cases": missing, "missing_evidence": absent}
    if any(row["frames"] != config["calibration_frames"] for row in rows):
        raise ValueError("native points differ from the frozen Calibration cohort")
    return decide(directory, config, points, rows)


def decide(directory: Path, config: dict, points: list[dict], rows: list[dict]) -> dict:
    frames = config["calibration_frames"]
    baseline = fully_measured_control(json_at(directory / "legacy/a4/summary.json"), frames)
    correct = json_at(directory / "native_correctness.json")["verdict"] == "PASS"
    candidates = [row for row in rows if row["model"] == "hts"]
    profiles, _, _ = compare_profiles(points, config)
    ld_gains = [row for profile in profiles.values() for rows in profile["hts_vs_ld_low_rate_matches"].values()
                for row in rows if row["rate_saving_percent"] > 0]
    old_gains = old_base_improvements(candidates, baseline, config["phase0"])
    passed = correct and bool(ld_gains or old_gains)
    report = {"verdict": "PASS" if passed else "FAIL", "native_correctness": correct,
              "native_cases_completed": len(rows), "matched_low_rate_gains_over_ld": ld_gains,
              "matched_low_rate_gains_over_a4_base": old_gains, "calibration_a4_base": baseline}
    if passed:
        choices = select_qp(candidates, config["rate_targets_mbps"])
        write_json(directory / "frozen_qp_points.json", choices)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = inspect(args.directory, json_at(CONFIG))
    write_json(args.directory / "phase_verdict.json", report)
    print(report["verdict"])
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
