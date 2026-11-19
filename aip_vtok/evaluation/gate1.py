"""Fail-closed Gate 1 decision and merged dual-seed machine-readable evidence."""

import csv
import json

from aip_vtok.evaluation.smoke import BUDGETS
from aip_vtok.runtime import resolve, write_json
from aip_vtok.training.formal import settings


def monotonic(metrics, limits):
    comparisons = []
    for lower, upper in zip(BUDGETS, BUDGETS[1:]):
        a, b = metrics[str(lower)], metrics[str(upper)]
        gain = b["psnr_global_mse"] - a["psnr_global_mse"]
        required = limits["core_psnr_gain_db"] if (lower, upper) in ((1024, 2048), (2048, 4096)) else 0
        comparisons.append(
            gain > 0 and gain >= required and b["lpips"] < a["lpips"] and b["dists"] < a["dists"]
        )
    return all(comparisons)


def seed_checks(seed, limits):
    checks = {
        "training_complete": seed["training_steps"] >= 1000,
        "exact_independent_loops": seed["maximum_independent_rgb_error"] == 0
        and seed["independent_ids_all_equal"],
        "healthy_fsq": all(
            p["minimum_unique_ids"] > 1
            and p["minimum_dimension_entropy_bits"] >= limits["minimum_dimension_entropy_bits"]
            for p in seed["health_summary"].values()
        ),
        "content_interventions": all(p["ci95_low"] > 0 for p in seed["intervention_summary"].values()),
    }
    for mode, metrics in seed["summary"].items():
        checks[f"{mode}_174_frames"] = all(metrics[str(k)]["frames"] == 174 for k in BUDGETS)
        checks[f"{mode}_monotonic_core_gains"] = monotonic(metrics, limits)
        checks[f"{mode}_absolute_quality"] = all(
            metrics[str(k)]["psnr_global_mse"] >= limits["minimum_high_budget_psnr_db"] for k in (4096, 8192)
        )
        checks[f"{mode}_chroma"] = all(
            metrics[str(k)]["chroma_retention"] >= limits["minimum_chroma_retention"] for k in (4096, 8192)
        )
    return checks


def assess_gate1(seeds, limits):
    complete = {seed["seed"] for seed in seeds} == {1, 2}
    checks = {str(seed["seed"]): seed_checks(seed, limits) for seed in seeds}
    passed = complete and all(all(values.values()) for values in checks.values())
    return {
        "gate": 1,
        "passed": passed,
        "both_seeds_complete": complete,
        "checks": checks,
        "limits": limits,
        "decision": "allow_gate2" if passed else "stop_at_gate1",
        "gate2_allowed": passed,
    }


def build_gate1():
    config = settings("configs/aip_vtok/gate1.yaml", 1)
    directory = resolve(config["output"])
    paths = sorted(directory.glob("gate1_seed*/evaluation.json"))
    seeds = [json.loads(path.read_text()) for path in paths]
    result = assess_gate1(seeds, config["stage1"]["limits"])
    result["seed_evidence"] = seeds
    write_json(directory / "gate1_decision.json", result)
    write_json(
        directory / "gate1_interventions.json", {str(s["seed"]): s["intervention_summary"] for s in seeds}
    )
    rows = []
    for path in paths:
        with (path.parent / "frame_metrics.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    if rows:
        with (directory / "gate1_multi_budget_rd.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


if __name__ == "__main__":
    result = build_gate1()
    print(json.dumps({k: result[k] for k in ("passed", "both_seeds_complete", "checks")}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)
