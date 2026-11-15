"""Longer multi-scene representation training after short diagnostics; all outputs stay separate."""

import argparse
import copy

from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/optimization_20261115/d1/config.json")
    parser.add_argument("--checkpoint", default="checkpoints/optimization_20261115/d1/latest.pt")
    parser.add_argument("--output", default="results/optimization_20261115/multiscene_extended")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    config = copy.deepcopy(read_config(args.source))
    config["purpose"] = "train"
    config["paths"].update(
        checkpoint=args.checkpoint, output=args.output, save_checkpoint=args.output + "/latest.pt"
    )
    train = config["training"]
    train.update(
        query_initialization_scale=1.0,
        grid_query_initialization_scale=1.0,
        lr=0.0001,
        save_every=100,
        validate_every=1000000,
        crop=[1080, 1920],
        sequence_length=4,
        sequence_schedule=[],
        resolution_schedule=[],
        diagnostic_steps=[0, 10, 50, 100],
        diagnostic_every=100,
    )
    train["vq"].update(warmup_steps=0, initialization=None, strategy="deterministic_nearest")
    config["teacher"] = {"policy": "i_only", "decay_steps": args.steps, "minimum_factor": 0.05}
    write_json(args.output + "/config.json", config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=args.checkpoint)
    run_training(model, config, args.steps, status="trained")


if __name__ == "__main__":
    main()
