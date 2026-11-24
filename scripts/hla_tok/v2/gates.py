"""Gate decisions derived from frozen, sample-matched experiment artifacts."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.v2.protocol import OUTPUT, csv_at, json_at

ARMS = ("R0", "R1", "R2", "R3")
QUALITY = ("global_psnr", "lpips", "dists")


def differences(row, control):
    return {
        "psnr_gain_db": float(row["global_psnr"]) - float(control["global_psnr"]),
        "lpips_reduction": float(control["lpips"]) - float(row["lpips"]),
        "dists_reduction": float(control["dists"]) - float(row["dists"]),
    }


def overfit():
    output = OUTPUT / "diag1_overfit"
    rows = []
    for arm in ARMS:
        for stage in ("rgb", "overfit"):
            folder = output / arm / stage
            curve = csv_at(folder / "training_curve.csv")
            head, tail = curve[0], curve[-1]
            rows.append({"arm": arm, "loss_arm": stage, "step": int(tail["step"]),
                         **{key: float(tail[key]) for key in (*QUALITY, "effective_rank", "centered_variance")},
                         **differences(tail, head),
                         "parameter_update_l2": json_at(folder / "training.json")["parameter_update_l2"]})
    successful = [r["arm"] for r in rows if r["loss_arm"] == "rgb" and r["psnr_gain_db"] >= 3.0]
    write_csv(output / "metrics.csv", rows)
    return {"status": "PASS" if successful else "FAIL", "single_sample_fit_arms": successful,
            "interpretation": "A 3 dB paired single-sample PSNR gain establishes a roughly twofold MSE reduction."}


def load_arm(seed, arm):
    folder = OUTPUT / f"continuous_pilot/seed{seed}/{arm}/pilot"
    row = json_at(folder / "validation/summary.json")
    run = json_at(folder / "training.json")
    init = json_at(folder / "initialization.json")
    return row | {"arm": arm, "seed": seed, "sample_chain": run["sample_chain"], "seen_pixels": run["seen_pixels"],
                  "steps": run["optimizer_updates"], "shared_init_sha256": init["shared_sha256"],
                  "checkpoint": json_at(folder / "latest.json")["path"]}


def comparable(arms):
    keys = ("sample_chain", "seen_pixels", "steps", "shared_init_sha256", "frames")
    if len({tuple(r[k] for k in keys) for r in arms}) != 1:
        raise ValueError("architecture comparison must match the shared initialization and actual data trajectory")


def screen(row):
    return row["psnr_gain_db"] >= 2 and row["lpips_reduction"] >= 0.10 and row["dists_reduction"] > 0


def pick(rows):
    eligible = {r["arm"]: r for r in rows if r["arm"] in {"R2", "R3"} and screen(r)}
    if "R2" not in eligible:
        return eligible.get("R3")
    return eligible["R3"] if "R3" in eligible and all(
        eligible["R3"][name] > eligible["R2"][name] for name in
        ("lpips_reduction", "dists_reduction", "psnr_gain_db")
    ) else eligible["R2"]


def pilot(seed, candidate, extension=False):
    output = OUTPUT / f"continuous_pilot/seed{seed}"
    if extension:
        output /= f"extension_{candidate}"
    names = ARMS if seed == 1 and not extension else ("R0", candidate)
    rows = [load_arm(seed, arm) for arm in names]
    comparable(rows)
    control = rows[0]
    for row in rows:
        row.update(differences(row, control))
        row["screen_pass"] = screen(row)
    write_csv(output / "metrics.csv", rows)
    selected = pick(rows) if seed == 1 else rows[-1]
    status = bool(selected) and (seed == 1 or all(selected[k] > 0 for k in
                                      ("psnr_gain_db", "lpips_reduction", "dists_reduction")))
    return {"status": "PASS" if status else "FAIL",
            "candidate": selected["arm"] if selected else None,
            "paired_comparisons": rows,
            "seed": seed,
            "reason": "Short-screen initial thresholds" if seed == 1 else "Held-out seed signs on matched data"}


def pilot_gate_path(pilot, seed):
    folder = pilot / f"seed{seed}"
    extensions = list(folder.glob("extension_*/acceptance.json"))
    if len(extensions) > 1:
        raise RuntimeError("only one paired extension is allowed")
    return extensions[0] if extensions else folder / "acceptance.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("overfit", "pilot"))
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--candidate", choices=("R2", "R3"))
    parser.add_argument("--extension-pair", action="store_true")
    args = parser.parse_args()
    if args.stage == "overfit":
        result, path = overfit(), OUTPUT / "diag1_overfit/acceptance.json"
    else:
        if (args.extension_pair or args.seed == 2) and args.candidate is None:
            raise ValueError("paired pilot requires an anchored candidate")
        output = OUTPUT / f"continuous_pilot/seed{args.seed}"
        if args.extension_pair:
            output /= f"extension_{args.candidate}"
        result, path = pilot(args.seed, args.candidate, args.extension_pair), output / "acceptance.json"
    write_json(path, result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
