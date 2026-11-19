#!/usr/bin/env python3
"""Run the gated dense C1-A smoke and two-seed pilot; continue atomic checkpoints."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.c1.artifacts import experiment  # noqa: E402
from aip_vtok.c1.pilot import run_a  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "smoke"), default="all")
    with experiment():
        run_a(parser.parse_args().stage)
        from aip_vtok.c1.report import generate

        generate()
