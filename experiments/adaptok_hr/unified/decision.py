"""Predeclared decisions use full-resolution Validation, never training-view LPIPS."""

import argparse
import json
import math

from adaptok_hr.runtime import stable_hash, write_json

from .evaluate import read_rows
from .run import ARCHITECTURES, ARMS, ROOT, checkpoint

BUDGETS = ("256", "512", "1024", "2048")


def mean_lpips(summary):
    return sum(summary["closed_loop"][k]["lpips"] for k in BUDGETS) / len(BUDGETS)


def protected_budgets(candidate, control, psnr_drop=.05, perceptual_increase=.005):
    result = {}
    for k in BUDGETS:
        a, b = candidate["closed_loop"][k], control["closed_loop"][k]
        result[k] = {"psnr": a["psnr_global"] >= b["psnr_global"] - psnr_drop,
                     "lpips": a["lpips"] <= b["lpips"] + perceptual_increase,
                     "dists": a["dists"] <= b["dists"] + perceptual_increase}
    return result


def all_protected(checks):
    return all(all(row.values()) for row in checks.values())


def pilot_arm_outcome(arm, measurements):
    per_seed = {}
    for seed, runs in measurements.items():
        control, candidate = runs["U0"], runs[arm]
        delta = mean_lpips(candidate) - mean_lpips(control)
        protected = protected_budgets(candidate, control)
        per_seed[seed] = {"mean_lpips": mean_lpips(candidate), "mean_lpips_delta": delta,
                          "improves": delta < 0, "protected": protected,
                          "protected_pass": all_protected(protected)}
    return {"seeds": per_seed, "mean_lpips": sum(r["mean_lpips"] for r in per_seed.values()) / 2,
            "both_improve": all(r["improves"] for r in per_seed.values()),
            "protected_pass": all(r["protected_pass"] for r in per_seed.values())}


def pilot_decision(measurements):
    outcomes = {arm: pilot_arm_outcome(arm, measurements) for arm in ("U1", "U2", "U3")}
    eligible = [arm for arm, row in outcomes.items() if row["both_improve"]] or list(outcomes)
    winner = min(eligible, key=lambda arm: (outcomes[arm]["mean_lpips"], arm))
    chosen = outcomes[winner]
    return {"winner": winner, "outcomes": outcomes,
            "unified_improves_both_seeds": chosen["both_improve"],
            "preferred_training_gate_pass": chosen["both_improve"] and chosen["protected_pass"],
            "preferred_action": "optimize_training" if chosen["both_improve"] else "analyze_cross_scale",
            "scope": "300-update D3 fine-tuning only; no claim about training from scratch"}


def scratch_quality_gate(measurements):
    outcomes = {}
    for arm in ("N1", "N2"):
        checks = {}
        for seed, runs in measurements.items():
            candidate, control = runs[arm], runs["N0"]
            protected = protected_budgets(candidate, control)
            checks[seed] = {"protected": protected, "protected_pass": all_protected(protected),
                            "mean_lpips_improves": mean_lpips(candidate) < mean_lpips(control),
                            "same_state_psnr_positive": candidate["same_state_gain_db"] > 0,
                            "same_state_lpips_positive": candidate["same_state_lpips_gain"] > 0,
                            "fine_shuffle_positive": candidate["full_fine"]["shuffle"]["mse_increase"] > 1e-8,
                            "fine_cross_positive": candidate["full_fine"]["cross"]["mse_increase"] > 1e-8}
        passed = all(all(value for key, value in row.items() if key != "protected") for row in checks.values())
        outcomes[arm] = {"pass": passed, "seeds": checks}
    return {"advance": any(row["pass"] for row in outcomes.values()), "outcomes": outcomes,
            "action": "continue_predeclared_milestones" if any(row["pass"] for row in outcomes.values())
                      else "archive_negative_and_stop"}


def training_stability(name, step):
    rows = read_rows(ROOT / name / "training.jsonl")
    active = [r for r in rows if r["step"] <= step]
    final = active[-1]["frame_metrics"]
    return {"updates": len(active), "finite_gradients": all(math.isfinite(r["gradient_norm"]) for r in active),
            "final_no_collapse": all(r["maximum_code_fraction"] <= .95 for r in final),
            "positive_vq": all(math.isfinite(r["vq_loss"]) and r["vq_loss"] > 0 for r in final),
            "positive_level_variance": all(level["feature_variance"] > 0 for row in final for level in row["levels"].values())}


def common_schedule(row):
    return {"input": row["input"], "main": row["budgets"],
            "crops": [r["perceptual_crops"] for r in row["frame_metrics"]]}


def matched_logs(seed, arms, begin, step):
    logs = {arm: [r for r in read_rows(ROOT / f"S{seed}_{arm}/training.jsonl") if begin < r["step"] <= step]
            for arm in arms}
    for rows in logs.values():
        if [r["step"] for r in rows] != list(range(begin + 1, step + 1)):
            raise ValueError("unmatched optimizer update count")
    return logs


def check_frequent_pairs(logs):
    for normal, frequent in zip(logs["U2"], logs["U3"], strict=True):
        for a, b in zip(normal["frame_metrics"], frequent["frame_metrics"], strict=True):
            if a["auxiliary_budget"] is not None and a["auxiliary_budget"] != b["auxiliary_budget"]:
                raise ValueError("frequent pairing changed an existing pair")


def auxiliary_schedules(logs, arms, stage):
    comparisons = list(arms[:3]) if stage == "A" else list(arms)
    schedules = {arm: stable_hash([[r["auxiliary_budget"] for r in row["frame_metrics"]] for row in logs[arm]])
                 for arm in comparisons}
    if len(set(schedules.values())) != 1:
        raise ValueError("control auxiliary schedules differ")
    if stage == "A":
        check_frequent_pairs(logs)
    return schedules


def match_seed(stage, seed, arms, begin, step):
    logs = matched_logs(seed, arms, begin, step)
    schedules = {arm: stable_hash([common_schedule(r) for r in rows]) for arm, rows in logs.items()}
    if len(set(schedules.values())) != 1:
        raise ValueError("data/main-budget/crop schedules differ")
    auxiliaries = auxiliary_schedules(logs, arms, stage)
    source = {arm: json.loads(checkpoint(f"S{seed}_{arm}", step).with_suffix(".json").read_text())["source_sha256"] for arm in arms}
    if len(set(source.values())) != 1:
        raise ValueError("production code differs across arms")
    return {"common_schedules": schedules, "auxiliary_schedules": auxiliaries,
            "source_identities": source, "additional_updates": step - begin,
            "frames": sum(r["frames"] for r in logs[arms[0]])}


def match_training(stage, step):
    arms, begin = (ARMS, 1300) if stage == "A" else (ARCHITECTURES, 0)
    proof = {str(seed): match_seed(stage, seed, arms, begin, step) for seed in (1, 2)}
    write_json(ROOT / f"match_{stage}_{step}.json", proof)
    return proof


def read_measurements(arms, step, frames):
    return {str(seed): {arm: json.loads((ROOT / f"S{seed}_{arm}/validation_{step}_{frames}/summary.json").read_text())
                        for arm in arms} for seed in (1, 2)}


def stability_screen(step):
    rows = {f"S{s}_{a}": training_stability(f"S{s}_{a}", step) for s in (1, 2) for a in ARCHITECTURES}
    passed = all(all(v for k, v in row.items() if k != "updates") for row in rows.values())
    return rows, passed


def relative_screen(measurements):
    rows = {f"S{s}_{a}": protected_budgets(runs[a], runs["N0"], .5, .03)
            for s, runs in measurements.items() for a in ("N1", "N2")}
    return rows, all(all_protected(r) for r in rows.values())


def scratch_decision(measurements, step):
    result = scratch_quality_gate(measurements)
    result["stability"], stable = stability_screen(step)
    if step == 300:
        result["screen"], result["relative_screen_pass"] = relative_screen(measurements)
        result["advance"] = all(row["finite_gradients"] for row in result["stability"].values())
    else:
        result["advance"] = result["advance"] and stable
    result["action"] = "continue_predeclared_milestones" if result["advance"] else "archive_negative_and_stop"
    return result


def decide(stage, step, frames):
    match_training(stage, step)
    measurements = read_measurements(ARMS if stage == "A" else ARCHITECTURES, step, frames)
    result = pilot_decision(measurements) if stage == "A" else scratch_decision(measurements, step)
    name = "decision_A.json" if stage == "A" else f"decision_B_{step}.json"
    output = ROOT / name
    write_json(output, result)
    print(json.dumps({"output": str(output), **{k: v for k, v in result.items() if k in ("winner", "advance", "preferred_action")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("A", "B"))
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    decide(args.stage, args.steps, args.frames)
