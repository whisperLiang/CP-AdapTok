#!/usr/bin/env python3
"""Run one stage of the sparse latent decoder pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.diagnostics.diag0.io import json_at  # noqa: E402
from aip_vtok.sparse_decoder.latency import run_latency  # noqa: E402
from aip_vtok.sparse_decoder.phase0 import run_phase0  # noqa: E402
from aip_vtok.sparse_decoder.phase1 import TESTS, require_phase1, run_phase1  # noqa: E402
from aip_vtok.sparse_decoder.pilot import run_evaluation  # noqa: E402
from aip_vtok.sparse_decoder.protocol import stage_audit  # noqa: E402
from aip_vtok.sparse_decoder.report import run_report  # noqa: E402
from aip_vtok.sparse_decoder.training import run_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"phase0": run_phase0, "train": run_training, "evaluate": run_evaluation, "latency": run_latency,
              "report": run_report}
    parser.add_argument("stage", choices=("preflight", "phase1", *stages))
    args = parser.parse_args()
    with stage_audit(args.stage, metrics=args.stage in {"preflight", "phase0", "evaluate"},
                     load=args.stage != "phase1") as run:
        if args.stage == "phase1":
            if TESTS.is_file() and json_at(TESTS)["exit_code"] == 0:
                require_phase1()
                print("Phase 1 already verified.", flush=True)
            else:
                run_phase1()
        elif args.stage == "preflight":
            print("Frozen inputs, GPU, metric weights, and W1 checkpoint verified.", flush=True)
        else:
            stages[args.stage](run)


if __name__ == "__main__":
    main()
