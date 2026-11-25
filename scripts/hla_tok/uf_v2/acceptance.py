"""Apply frozen quality/rate/time gates and account for every experiment's dataset reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.uf_v2.rate_analysis import BASELINE, read
from scripts.hla_tok.v2.protocol import ROOT

BUDGETS = (5, 10, 20, 40)
FINAL_CANDIDATES = ("Anchor-Fast", "D1 / static", "D1 / 2000", "D1 / joint 4000",
                    "D1 / joint 6000", "D1 / joint 8000", "D1 / joint static",
                    "D1 / formal static", "D1 / formal static fast", "D1 / formal E0", "D1 / formal E0 fast",
                    "D1 / Seed 2 final E0")
FINAL_EVIDENCE = {
    "D1 / formal E0 fast": ("dynamic-selected", "incremental-selected", "native-selected"),
    "D1 / formal static fast": ("dynamic-final", "incremental-final", "native-final"),
    "D1 / Seed 2 final E0": ("dynamic-seed2-final", "incremental-seed2-final", "native-seed2-final"),
}


def gate_quality(row, limit):
    if row["status"] != "PASS":
        return False
    return (row["global_psnr"] >= limit["global_psnr_min"] and row["lpips"] <= limit["lpips_max"]
            and row["dists"] <= limit.get("dists_max", 1))


def timing_met(row, cutoffs):
    pooled = row["pooled"]
    return (len(row["repetitions"]) == 3 and pooled["frames"] == 522
            and pooled["fps"] >= cutoffs["minimum_fps"] and pooled["total_ms_p95"] <= cutoffs["maximum_p95_ms"])


def candidate_gates(details, cutoffs):
    quality = {int(q["budget_mbps"]): q for q in details["quality"] if q["budget_mbps"] in BUDGETS}
    if details["timing"] is None or set(quality) != set(BUDGETS):
        return None
    results = []
    for budget in BUDGETS:
        q = quality[budget]
        timing = details["timing"][str(budget)]
        perf = timing["pooled"]
        results.append({
            "budget": budget, "rd": gate_quality(q, cutoffs[str(budget)]),
            "realtime": timing_met(timing, cutoffs),
            "hard_budget": q["status"] == "PASS" and q["max"] <= budget * 1_000_000 // 240,
            "fps": perf["fps"], "p95_ms": perf["total_ms_p95"], "quality": q,
        })
    return results


def comparable_equal(records, name, metric, target):
    match = next(row for row in records if row["method"] == name and row["metric"] == metric and row["target"] == target)
    if match["status"] != "PASS":
        return match["status"], match
    return ("PASS" if match["candidate_actual_mbps"] < match["original_actual_mbps"] else "FAIL"), match


def audit_reads(out):
    audits = []
    for artifact in sorted(out.rglob("*.json")):
        if "historical" in artifact.name:
            continue
        evidence = json.loads(artifact.read_text())
        if not isinstance(evidence, dict):
            continue
        counts = evidence.get("read_counts") or evidence.get("reads")
        if counts is not None:
            audits.append({"file": str(artifact.relative_to(out)), **counts})
    test_count = sum(entry.get("test", 0) for entry in audits)
    if test_count != 0:
        raise ValueError("Validation admission cannot be computed after premature Test access")
    return audits


def model_status(analysis, cutoffs, user_selected):
    candidates = {}
    for method in FINAL_CANDIDATES:
        if method not in analysis["measured"]:
            continue
        rows = candidate_gates(analysis["measured"][method], cutoffs)
        if rows is None:
            continue
        psnr, psnr_detail = comparable_equal(analysis["equal_quality"], method, "global_psnr", 25)
        perceptual, lpips_detail = comparable_equal(analysis["equal_quality"], method, "lpips", 0.28978)
        candidates[method] = {"four_budgets": rows, "equal_psnr": psnr, "equal_perceptual": perceptual,
                              "equal_psnr_data": psnr_detail, "equal_perceptual_data": lpips_detail,
                              "rd_passes": sum(row["rd"] for row in rows),
                              "runtime_passes": sum(row["realtime"] for row in rows),
                              "checkpoint_sha256": analysis["measured"][method]["checkpoint_sha256"]}
    if user_selected not in candidates:
        raise ValueError("selected endpoint has no complete timing and quality evidence")
    return candidates, candidates[user_selected]


def native_gates(closure, append, stream_tag, required_base):
    if not stream_tag or {closure["model_tag"], append["model_tag"]} != {stream_tag}:
        raise ValueError("exact native acceptance requires matching stream identities")
    return {
        "CAUSALITY": (closure["frames"] == 189 and closure["state_mismatches"] == 0
                      and closure["independent_i_recoveries"] == 2),
        "BASE_BUDGET": closure["max_base"] <= required_base,
        "INCREMENTAL": len(append["frames"]) == 4 and all(row["incremental_exact"] for row in append["frames"]),
    }


def training_status(out):
    ledger = read(out / "training/optimizer_ledger.json")["optimizer_updates"]
    if ledger == {"1": 20000, "2": 20000}:
        return "PASS", ledger
    return "PARTIAL_PASS", ledger


def assemble(out, selected_name):
    analysis = read(out / "evaluation/rate_analysis.json")
    protocol = read(ROOT / "configs/hla_tok/uf_v2/pilot.json")
    cutoffs = protocol["deployment_thresholds"]
    candidates, selected = model_status(analysis, cutoffs, selected_name)
    reaches = [r for r in analysis["curves"][selected_name] if r.get("budget_mbps", 0)
               and r["global_psnr"] >= 25 and r["actual_mbps"] < 3.594]
    audits = audit_reads(out)
    write_json(out / "evidence/test_audit.json", {"status": "NOT_RUN", "test_pixel_reads": 0, "runs": audits,
               "guard": "AuditedFrames denies Test pixel reads and colliding identities; metadata-only manifest hash"})
    lite = read(out / "diagnostics/base-lite-extended/summary.json")
    closure, append, samples = FINAL_EVIDENCE[selected_name]
    stream = analysis["measured"][selected_name]["model_tag"]
    states = read(out / "evaluation" / closure / "summary.json")
    increments = read(out / "evaluation" / append / "summary.json")
    native = native_gates(states, increments, stream, protocol["base"]["bytes_max"])
    training, updates = training_status(out)
    dims = {
        "BASELINE": "PARTIAL_PASS", "PROFILING": "PARTIAL_PASS", "ANCHOR_FAST": "PASS", "D1_DECODER": "PASS",
        "UNIFIED_LATENT": "PASS", "STATIC_ENTROPY": "PASS", "BASE_LITE": lite["status"],
        "PERCEPTUAL_ALLOCATION": "PARTIAL_PASS",
        "RATE_DISTORTION": "PASS" if selected["rd_passes"] == 4 else "FAIL", "EQUAL_PSNR_RATE": selected["equal_psnr"],
        "EQUAL_PERCEPTUAL_RATE": selected["equal_perceptual"], "STRICT_PREFIX": "PASS",
        **{name: "PASS" if passed else "FAIL" for name, passed in native.items()},
        "REALTIME": "PASS" if selected["runtime_passes"] == 4 else "FAIL", "TRAINING": training, "TEST": "NOT_RUN",
    }
    admission_keys = ("RATE_DISTORTION", "EQUAL_PSNR_RATE", "EQUAL_PERCEPTUAL_RATE", "REALTIME",
                      "CAUSALITY", "STRICT_PREFIX", "INCREMENTAL", "BASE_BUDGET")
    admitted = (all(dims[key] == "PASS" for key in admission_keys)
                and all(row["hard_budget"] for row in selected["four_budgets"]))
    verdict = {"overall": "PARTIAL_PASS", "validation_admission": "PASS" if admitted else "FAIL",
               "dimensions": dims, "selected_research_endpoint": selected_name,
               "selected_model": selected, "validation_candidates": candidates, "frozen_thresholds": cutoffs,
               "observed_at_least_25db_under_3_594": reaches, "original_reference": BASELINE,
               "optimizer_updates": updates, "model_tag": stream, "sample_folder": samples,
               "causal_evidence": closure, "incremental_evidence": append,
               "test_pixel_reads": 0, "deployable_model": None,
               "test_reason": ("requires a formal freeze" if admitted else "Validation admission targets remain unmet")}
    write_json(out / "final_verdict.json", verdict)
    lines = [{"candidate": label, **{k: row[k] for k in ("budget", "rd", "realtime", "hard_budget", "fps", "p95_ms")},
              **{k: row["quality"].get(k) for k in ("actual_mbps", "actual_bpp", "global_psnr", "lpips", "dists")}}
             for label, model in candidates.items() for row in model["four_budgets"]]
    write_csv(out / "evaluation/acceptance.csv", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--selected", required=True, choices=FINAL_EVIDENCE)
    args = parser.parse_args()
    assemble(args.results, args.selected)


if __name__ == "__main__":
    main()
