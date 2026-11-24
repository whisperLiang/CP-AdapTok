"""At most one matched 1000-update extension, preserving the primary four-arm result."""

# ruff: noqa: E402
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.gates import differences
from scripts.hla_tok.v2.protocol import OUTPUT, ROOT, csv_at, json_at


def preserve(folder, frozen):
    frozen.mkdir(exist_ok=True)
    if not (frozen / "validation").exists():
        for name in ("training.json", "training_curve.csv", "latest.json"):
            shutil.copy2(folder / name, frozen / name)
        (folder / "validation").rename(frozen / "validation")


def condition(pilot, candidate):
    gate = json_at(pilot / "acceptance.json")
    if gate["status"] == "PASS":
        raise ValueError("the primary pilot already cleared the short screen")
    previous = {row["arm"]: row for row in gate["paired_comparisons"]}
    improvement = differences(previous[candidate], previous["R0"])
    curve = csv_at(pilot / candidate / "pilot/training_curve.csv")
    late = differences(curve[-1], curve[-2])
    consistent_progress = all(value > 0 for value in late.values())
    advantages = sum(value > 0 for value in improvement.values())
    if not consistent_progress or advantages < 2:
        raise RuntimeError("the observed trajectory does not support a bounded matched extension")
    return {"candidate": candidate, "status": "AUTHORIZED_BY_MEASURED_TREND",
            "max_extra_updates_per_arm": 1000, "paired_174_at_1000": improvement,
            "last_validation_subset_interval": late, "arms": ["R0", candidate]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=("R2", "R3"))
    args = parser.parse_args()
    pilot = OUTPUT / "continuous_pilot/seed1"
    if any(pilot.glob("extension_*/decision.json")):
        raise RuntimeError("the paired extension has already been scheduled")
    decision = condition(pilot, args.candidate)
    folders = [pilot / arm / "pilot" for arm in decision["arms"]]
    checkpoints = [json_at(folder / "latest.json") for folder in folders]
    if any(checkpoint["step"] != 1000 for checkpoint in checkpoints):
        raise ValueError("both arms must start exactly at the primary pilot boundary")
    target = pilot / f"extension_{args.candidate}"
    target.mkdir(exist_ok=False)
    write_json(target / "decision.json", decision)
    for arm, folder, initial in zip(decision["arms"], folders, checkpoints, strict=True):
        preserve(folder, folder / "first_1000")
        subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--stage", "pilot",
                        "--arm", arm, "--resume", str(ROOT / initial["path"]), "--extend"], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("gates.py")), "pilot",
                    "--candidate", args.candidate, "--extension-pair"], check=True)


if __name__ == "__main__":
    main()
