#!/usr/bin/env python3
"""Run the C1-Followup experiment from its predeclared Train-only calibration."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.c1_followup.experiment import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibrate", "replay", "validate", "train", "sparse", "report"),
                        default="all")
    main(parser.parse_args().stage)
