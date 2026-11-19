#!/usr/bin/env python3
"""Run the gated AIP-VTok Sparse-RFSQ experiment or regenerate its reports."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.sparse_rfsq.experiment import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibrate", "s1", "test", "s2", "report"), default="all")
    parser.add_argument("--wait-for-gpu", action="store_true", help="wait for other GPU processes to finish")
    options = parser.parse_args()
    main(options.stage, options.wait_for_gpu)
