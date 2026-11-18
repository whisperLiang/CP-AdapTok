"""Reproducible CLI for one matched optimizer run or a sequence of arms."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from adaptok_hr.runtime import read_config

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("results/adaptok_hr_fine_detail_v2")
CHECKPOINTS = Path("checkpoints/adaptok_hr_fine_detail_v2")
PROTOCOL = json.loads((PACKAGE / "protocol.json").read_text())


def run_name(seed, arm):
    return f"S{seed}_{arm}"


def checkpoint(name, step):
    return CHECKPOINTS / name / f"step_{step:06d}.pt"


def configuration(seed, arm):
    model_seed, data_seed, budget_seed = PROTOCOL["seeds"][seed - 1]
    mode, weight = PROTOCOL["arms"][arm]
    config = read_config(PACKAGE.parent / "fine_recovery/configs/A1.yaml")
    config["model"]["fine_input"] = mode
    config["training"].update(seed=model_seed, model_seed=model_seed, budget_seed=budget_seed,
                              extra_checkpoints=[10, 25])
    config["training"]["losses"]["suffix_detail"] = weight
    config["experiment"] = {"family": "fine_detail_v2", "arm": arm, "data_seed": data_seed,
                            "name": run_name(seed, arm), "initialization": "from_scratch"}
    return config


def execute(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", *map(str, arguments)]
    with log.open("a") as file:
        print(json.dumps(command), file=file, flush=True)
        subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def train(seed, arm, steps):
    name = run_name(seed, arm)
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / f"{name}.yaml"
    content = yaml.safe_dump(configuration(seed, arm), sort_keys=False)
    if config.exists() and config.read_text() != content:
        raise ValueError(f"existing configuration differs for {name}")
    config.write_text(content)
    path = checkpoint(name, steps)
    if path.is_file() and path.with_suffix(".json").is_file():
        return
    command = ["adaptok_hr.training.train", "--config", config, "--steps", steps,
               "--checkpoints", CHECKPOINTS, "--output", ROOT / name]
    committed = sorted(p for p in path.parent.glob("step_*.pt") if p.with_suffix(".json").exists())
    if committed:
        command.extend(["--resume", committed[-1]])
    execute(command, ROOT / "logs" / f"{name}_train_{steps}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--arms", nargs="+", choices=PROTOCOL["arms"], default=list(PROTOCOL["arms"]))
    parser.add_argument("--steps", type=int, choices=(300, 1000), default=300)
    args = parser.parse_args()
    for arm in args.arms:
        train(args.seed, arm, args.steps)


if __name__ == "__main__":
    main()
