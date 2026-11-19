#!/usr/bin/env python3
"""Run frozen Train calibration, matched probes and the fixed-reference rollout."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip_vtok.c1.artifacts import (  # noqa: E402
    LOCKED,
    audit,
    experiment,
    finish_audit,
    frozen_inputs,
    load_frozen,
    locked_artifacts,
)
from aip_vtok.c1.calibrate import calibrate, load_fits  # noqa: E402


def run(stage):
    if stage == "report":
        from aip_vtok.c1.report import generate

        generate()
        return
    record = audit(f"diag05_{stage}")
    config, identity = frozen_inputs()
    codec = load_frozen(config)
    if stage == "calibrate" or (stage == "all" and not LOCKED.is_file()):
        if LOCKED.is_file():
            raise SystemExit("Train selection is already locked in scan_lock.json")
        calibrate(config, identity, codec)
    if stage in ("all", "matched", "closed"):
        from aip_vtok.c1.diagnostic import evaluate

        lock, fit = locked_artifacts(identity)
        evaluate(config, codec, load_fits(fit), lock, stage)
    finish_audit(record)
    if stage == "all":
        from aip_vtok.c1.report import generate

        generate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibrate", "matched", "closed", "report"), default="all")
    with experiment():
        run(parser.parse_args().stage)


if __name__ == "__main__":
    main()
