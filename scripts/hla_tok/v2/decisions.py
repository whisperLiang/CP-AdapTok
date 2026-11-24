"""Record the conditional evidence gates after the matched pilot."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hla_tok.evaluation.io import write_json
from hla_tok.training.train import fixed_quality_gate
from scripts.hla_tok.v2.gates import QUALITY, differences, pilot_gate_path
from scripts.hla_tok.v2.protocol import OUTPUT, ROOT, csv_at, json_at


def confirmed_pilot():
    pilot = OUTPUT / "continuous_pilot"
    first = json_at(pilot_gate_path(pilot, 1))
    if first["status"] != "PASS":
        return first | {"status": "FAIL", "reason": "No anchored candidate cleared the immutable short-pilot screen"}
    candidate = first["candidate"]
    second = json_at(pilot_gate_path(pilot, 2))
    cases = csv_at(pilot / "seed1/interventions/metrics.csv")
    altered, = (row for row in cases if row["arm"] == candidate and row["intervention"] == "regional_mean")
    timings = csv_at(OUTPUT / f"benchmark/seed1/{candidate}/timings.csv")
    rates = {r["budget"]: float(r["neural_fps"]) for r in timings
             if r["precision"] == "bf16" and int(r["budget"]) in (8192, 16320)}
    intervention_effect = float(altered["global_psnr_delta"]) < 0 or float(altered["delta_dists"]) > 0
    stable = second["status"] == "PASS" and second["candidate"] == candidate
    realtime = len(rates) == 2 and all(v >= 30 for v in rates.values())
    return {"status": "PASS" if stable and intervention_effect and realtime else "FAIL",
            "candidate": candidate, "seed2_paired_direction": stable, "useful_token_intervention": intervention_effect,
            "neural_fps_bf16": rates, "intervention": altered, "checkpoint": next(
                row["checkpoint"] for row in first["paired_comparisons"] if row["arm"] == candidate)}


def quantized():
    arch = json_at(OUTPUT / "continuous_pilot/selection.json")["candidate"]
    output = OUTPUT / f"quantized/seed1/{arch}"
    measured = {arm: json_at(output / arm / "validation/summary.json") for arm in ("control", "quantized")}
    measured["bypass"] = json_at(output / "quantized/bypass_validation/summary.json")
    prior, = csv_at(ROOT / "results/hla_tok/phase1/fixed/validation_metrics.csv")
    improvements = differences(measured["quantized"], prior)
    enough = all(value > 0 for value in improvements.values())
    return {"status": "PASS" if enough else "FAIL", "candidate": arch,
            "vs_legacy_fixed": improvements, "quantized_vs_own_bypass": differences(measured["quantized"], measured["bypass"]),
            "quantized_vs_continuous_control": differences(measured["quantized"], measured["control"]),
            "continuous_control_vs_quantized_bypass": differences(measured["control"], measured["bypass"]),
            "checkpoint": json_at(output / "quantized/latest.json")["path"]}


def native():
    candidate = json_at(OUTPUT / "quantized/acceptance.json")["candidate"]
    output = OUTPUT / f"native/seed1/{candidate}/native"
    measured = json_at(output / "validation/summary.json")
    enough = measured["frames"] == 174 and measured["quantized"] and measured["rate_basis"] == "actual_fixed13_payload"
    return {"status": "PASS" if enough else "FAIL", "candidate": candidate,
            "fixed16320": {key: measured[key] for key in (*QUALITY, "actual_bpp")},
            "checkpoint": json_at(output / "latest.json")["path"]}


def final():
    trained = json_at(OUTPUT / "native/acceptance.json")
    arm = trained["candidate"]
    baseline = csv_at(ROOT / "results/hla_tok/baselines/metrics.csv")
    quality = fixed_quality_gate(trained["fixed16320"], baseline)
    times = csv_at(OUTPUT / f"benchmark/native/{arm}/timings.csv")
    rates = {r["budget"]: float(r["neural_fps"]) for r in times
             if r["precision"] == "bf16" and int(r["budget"]) in (8192, 16320)}
    fast = len(rates) == 2 and all(row >= 30 for row in rates.values())
    return {"status": "PASS" if quality["quality_pass"] and fast else "FAIL",
            "candidate": arm, "same_bpp_original_gate": quality, "neural_fps_bf16": rates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("D2", "D3", "D4", "D5"))
    args = parser.parse_args()
    name, func = {
        "D2": ("continuous_pilot/selection.json", confirmed_pilot), "D3": ("quantized/acceptance.json", quantized),
        "D4": ("native/acceptance.json", native), "D5": ("benchmark/acceptance.json", final),
    }[args.phase]
    outcome = func()
    write_json(OUTPUT / name, outcome)
    print(args.phase, outcome["status"], flush=True)


if __name__ == "__main__":
    main()
