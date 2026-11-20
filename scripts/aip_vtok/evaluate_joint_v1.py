"""Evaluate all eight prefixes on the sealed native Validation or held-out Test."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.joint.evaluation import claim_test, evaluate, load_selection
from aip_vtok.joint.protocol import CONFIG, OUTPUT, exclusive_run, open_experiment
from aip_vtok.runtime import write_json


def evaluate_seed(config, seed, split):
    with exclusive_run():
        run = open_experiment(config, seed)
        selection = load_selection(run, seed)
        directory = OUTPUT / f"seed{seed}" / split
        seal = claim_test(run, selection) if split == "test" else None
        try:
            evaluate(run, split, directory, fixed=split == "validation", compare=split == "validation" and seed == 1,
                     archive=True, extend=split == "validation")
        except BaseException:
            if seal is not None:
                write_json(OUTPUT / "test_release.json", seal | {"status": "FAILED", "pixel_reads": run.data.counts})
            raise
        if seal is not None:
            write_json(OUTPUT / "test_release.json", seal | {"status": "TESTED_ON_HELDOUT", "pixel_reads": run.data.counts})
        if seed == 1:
            files = [f"{split}_metrics.csv", f"{split}_per_frame.csv"]
            if split == "validation":
                files += ["latency_breakdown.csv", "rate_ledger.csv", "closed_loop_metrics.csv"]
            for name in files:
                shutil.copyfile(directory / name, OUTPUT / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    evaluate_seed(args.config, args.seed, args.split)


if __name__ == "__main__":
    main()
