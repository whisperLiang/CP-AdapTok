#!/usr/bin/env python3
"""Run one stage of the sparse latent decoder pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.sparse_decoder.phase0 import run_phase0  # noqa: E402
from aip_vtok.sparse_decoder.protocol import stage_audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "phase0"))
    args = parser.parse_args()
    with stage_audit(args.stage) as run:
        if args.stage == "phase0":
            run_phase0(run)
        else:
            print("Frozen inputs, GPU, metric weights, and W1 checkpoint verified.", flush=True)


if __name__ == "__main__":
    main()
