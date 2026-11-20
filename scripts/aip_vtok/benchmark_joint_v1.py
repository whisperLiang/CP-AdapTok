"""Benchmark hardware latency and wire bytes for each native Validation prefix."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.joint.evaluation import evaluate, load_selection
from aip_vtok.joint.protocol import CONFIG, OUTPUT, exclusive_run, open_experiment


def benchmark_seed(config, seed):
    with exclusive_run():
        run = open_experiment(config, seed)
        load_selection(run, seed)
        evaluate(run, "validation", OUTPUT / f"seed{seed}" / "benchmark", metrics=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    benchmark_seed(args.config, args.seed)


if __name__ == "__main__":
    main()
