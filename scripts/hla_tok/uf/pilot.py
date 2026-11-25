"""Admit the next Anchor stage from paired Calibration evidence, never from Test."""

import argparse
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import csv_at, json_at


def paired_gains(frames: list[dict]) -> dict:
    clips = defaultdict(dict)
    for row in frames:
        clips[row["video"]][(int(row["frame"]), int(row["budget_mbps"]))] = row
    gains = {}
    for video, entries in clips.items():
        base = {frame: row for (frame, budget), row in entries.items() if budget == 0}
        gains[video] = {
            budget: {key: mean(float(entries[frame, budget][key]) - float(source[key])
                               for frame, source in base.items()) for key in ("mse", "lpips", "dists")}
            for budget in (5, 10)
        }
    return gains


def gradient_check(rows: list[dict], steps: int) -> bool:
    if [int(row["step"]) for row in rows] != list(range(1, steps + 1)):
        return False
    values = [(float(row["loss"]), float(row["gradient_norm"])) for row in rows]
    return all(isfinite(loss) and isfinite(gradient) and gradient > 0 for loss, gradient in values)


def admit(directory: Path, training: Path, audit: Path, target: int) -> dict:
    training_steps = 500 if target == 2000 else 2000
    spec = json_at(directory / "summary.json")
    manifest = json_at(training / f"step{training_steps}.json")
    untouched = spec["read_counts"]["test"] == manifest["read_counts"]["test"] == 0
    score = {row["budget_mbps"]: row for row in spec["rows"]}
    complete = all(row["status"] == "PASS" and row["base_oversized_frames"] == 0 for row in score.values())
    gradients = gradient_check(csv_at(training / "training.csv"), training_steps)
    native = json_at(audit)
    base_exact = native["frames"] == native["nal_bytes_matches"] and native["latent_matches"] > 0
    gains = paired_gains(csv_at(directory / "per_frame.csv")) if complete else {}
    consistent = all(sum(result[budget]["mse"] < 0 and result[budget]["lpips"] < 0 for result in gains.values())
                     > len(gains) // 2 for budget in (5, 10))
    requirements = {"closed_loop_and_cap": complete, "native_parity": base_exact, "training_gradients": gradients,
                    "repeated_calibration_gain": consistent, "test_sealed": untouched}
    return {"status": "PASS" if all(requirements.values()) else "FAIL", "next_optimizer_step_target": target,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "requirements": requirements, "paired_gain": gains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = admit(args.calibration, args.training, args.native_audit, 2000)
    write_json(args.output, result)
    if result["status"] != "PASS":
        raise SystemExit("A2 admission was not met; see the saved evidence")


if __name__ == "__main__":
    main()
