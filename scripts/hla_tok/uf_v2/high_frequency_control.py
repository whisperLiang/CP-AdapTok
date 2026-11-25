"""Run the matched 2,000-update P2 control with only the high-frequency weight set to zero."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.uf_v2 import train_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(train_joint.CONFIG.read_text())
    config["weights"]["high_frequency"] = 0.0
    override = args.output / "joint-no-high-frequency.json"
    write_json(override, config)
    trainer_arguments = ["train_joint", "--weights", str(args.weights), "--resume", str(args.resume), "--seed", "1",
                         "--steps", "4000", "--checkpoints", str(args.checkpoints), "--output", str(args.output)]
    with patch.object(train_joint, "CONFIG", override), patch.object(sys, "argv", trainer_arguments):
        train_joint.main()


if __name__ == "__main__":
    main()
