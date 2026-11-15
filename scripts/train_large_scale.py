"""Reproducible shared-S0 / B0 / B1 training. --max-steps remains an absolute phase target."""

import argparse
import fcntl
import json

from cp_adaptok.large_scale import PHASES, experiment, run_all, run_phase, run_preflight
from cp_adaptok.runtime import MissingPrerequisites, resolve, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", *PHASES, "evaluate"))
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--branch", choices=("both", "B0", "B1"), default="both")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.run_all and args.phase:
        parser.error("use either --phase or --run-all")
    if not args.phase and not args.run_all and not args.resume:
        parser.error("choose --phase, --run-all, or --resume")
    root = resolve("results/large_scale")
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / "training.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise MissingPrerequisites(["another large-scale scheduler is already running"])
            if args.phase == "preflight":
                run_preflight()
                return 0
            manifest, splits = experiment(args.run_id, args.seed)
            phase = args.phase or manifest.get("current", {}).get("phase")
            if args.run_all:
                run_all(manifest, splits, args.resume)
                from cp_adaptok.large_scale_analysis import evaluate_experiment

                evaluate_experiment(manifest, splits)
            elif phase == "evaluate":
                from cp_adaptok.large_scale_analysis import evaluate_experiment

                evaluate_experiment(manifest, splits)
            else:
                if phase is None:
                    parser.error("no phase to resume in this experiment")
                branches = ("shared",) if phase in ("pilot", "shared_warmup") else (
                    ("B0", "B1") if args.branch == "both" else (args.branch,))
                for branch in branches:
                    run_phase(phase, branch, manifest, splits, args.resume)
            print(json.dumps({"run_id": manifest["run_id"], "status": "requested_phases_finished"}), flush=True)
            return 0
    except MissingPrerequisites as error:
        write_json(root / "blocked.json", {"conditions": error.conditions, "completed": False})
        print(str(error), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
