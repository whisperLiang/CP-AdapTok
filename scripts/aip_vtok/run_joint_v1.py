"""Train or resume the AIP-VTok Joint V1 experiment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.joint.protocol import CHECKPOINTS, CONFIG, OUTPUT, exclusive_run, open_experiment
from aip_vtok.joint.report import figures, optional_json, write_report
from aip_vtok.joint.trainer import train_experiment


def train_seed(config, seed, resume):
    with exclusive_run():
        run = open_experiment(config, seed)
        train_experiment(run, seed, resume=resume)


def pipeline(config):
    from scripts.aip_vtok.benchmark_joint_v1 import benchmark_seed
    from scripts.aip_vtok.evaluate_joint_v1 import evaluate_seed

    for seed in (1, 2):
        checkpoint = CHECKPOINTS / f"seed{seed}" / "index.json"
        root = OUTPUT / f"seed{seed}"
        status = optional_json(OUTPUT / f"train_seed{seed}.json")
        if status.get("status") != "TRAINED":
            train_seed(config, seed, resume=checkpoint.is_file())
        if not (root / "validation" / "evaluation.json").is_file():
            evaluate_seed(config, seed, "validation")
    seal = optional_json(OUTPUT / "test_release.json")
    if not seal:
        evaluate_seed(config, 1, "test")
    elif seal["status"] != "TESTED_ON_HELDOUT":
        raise RuntimeError("The single sealed Test attempt failed or was interrupted")
    if not (OUTPUT / "seed1/benchmark/evaluation.json").is_file():
        benchmark_seed(config, 1)
    figures()
    write_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--pipeline", action="store_true", help="run both seeds, native evaluations, and final report")
    mode.add_argument("--report", action="store_true", help="render the current report and available figures")
    args = parser.parse_args()
    if args.pipeline:
        if args.seed != 1:
            parser.error("--pipeline starts with seed 1")
        pipeline(args.config)
    elif args.report:
        figures()
        write_report()
    else:
        train_seed(args.config, args.seed, args.resume)


if __name__ == "__main__":
    main()
