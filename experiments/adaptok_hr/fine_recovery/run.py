"""Serial, restartable matched experiments with optimizer-boundary recovery."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from adaptok_hr.codec.codec import tensor_digest
from adaptok_hr.runtime import write_json

ROOT = Path("results/adaptok_hr_fine_recovery")
CHECKPOINTS = Path("checkpoints/adaptok_hr_fine_recovery")
CONFIGS = Path("experiments/adaptok_hr/fine_recovery/configs")


def train(name, steps):
    output = ROOT / name
    command = [sys.executable, "-m", "adaptok_hr.training.train", "--config", str(CONFIGS / f"{name}.yaml"),
               "--output", str(output), "--checkpoints", str(CHECKPOINTS), "--steps", str(steps)]
    checkpoints = sorted((CHECKPOINTS / name).glob("step_*.pt"))
    if checkpoints:
        if (CHECKPOINTS / name / f"step_{steps:06d}.pt").exists():
            return
        command += ["--resume", str(checkpoints[-1])]
    execute(command, ROOT / f"{name}_train_{steps}.log")


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("a") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                       env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, check=True)


def evaluate(name, steps, frames):
    output = ROOT / f"{name}/validation_{steps:06d}_{frames}f"
    if (output / "summary.json").exists():
        return
    execute([sys.executable, "-m", "experiments.adaptok_hr.fine_recovery.evaluate", "--checkpoint",
             str(CHECKPOINTS / name / f"step_{steps:06d}.pt"), "--output", str(output), "--frames", str(frames)],
            ROOT / f"{name}_validation_{steps}_{frames}f.log")


def verify_matching(names, steps):
    rows = {name: [json.loads(line) for line in (ROOT / name / "training.jsonl").read_text().splitlines()
                   if json.loads(line)["step"] <= steps] for name in names}
    first = rows[names[0]]
    reference = [(r["input"]["plan"], r["budgets"], [f["auxiliary_budget"] for f in r["frame_metrics"]]) for r in first]
    for name, records in rows.items():
        actual = [(r["input"]["plan"], r["budgets"], [f["auxiliary_budget"] for f in r["frame_metrics"]]) for r in records]
        if actual != reference or len(records) != steps:
            raise ValueError(f"unmatched data/budget/update sequence: {name}")
    initialization, settings, sources = {}, {}, {}
    for name in names:
        initial = torch.load(CHECKPOINTS / name / "step_000000.pt", map_location="cpu", weights_only=False)
        common = {k: v for k, v in initial["model"].items() if not k.startswith("vq.")}
        initialization[name] = tensor_digest(common).hex()
        settings[name] = initial["config"]["training"]
        sources[name] = json.loads((ROOT / name / "run.json").read_text())["environment"]["source_sha256"]
    if len(set(initialization.values())) != 1 or any(s != settings[names[0]] for s in settings.values()):
        raise ValueError("unmatched common initial weights or training settings")
    if len(set(sources.values())) != 1:
        raise ValueError("training implementation changed between matched runs")
    write_json(ROOT / f"matching_{steps:06d}.json", {"names": names, "completed_steps": steps,
               "training_frames_each": sum(r["frames"] for r in first), "data_and_budget_sequences_identical": True,
               "encoder_decoder_tokenizer_initial_sha256": initialization, "training_source_sha256": sources,
               "training_settings_identical": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "match"))
    parser.add_argument("--names", nargs="+", default=["A0", "A1", "A2", "A3"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "match":
        verify_matching(args.names, args.steps)
        return
    for name in args.names:
        if args.mode == "train":
            train(name, args.steps)
        else:
            evaluate(name, args.steps, args.frames)


if __name__ == "__main__":
    main()
