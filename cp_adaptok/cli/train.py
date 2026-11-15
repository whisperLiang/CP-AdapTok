import argparse
import copy

from cp_adaptok.runtime import (
    MissingPrerequisites,
    build_model,
    preflight,
    read_config,
    report_missing,
    resolve,
    seed_everything,
)
from cp_adaptok.training import run_training


def apply_smoke_config(config):
    config["training"].update(crop=64, sequence_length=2, tbptt=2, sequence_schedule=[])
    config["loss"].update(latent=0, lpips=0, dists=0)
    config["paths"]["output"] = f"results/s{config['stage']}_smoke"
    config["paths"]["save_checkpoint"] = f"checkpoints/s{config['stage']}_smoke/latest.pt"
    if config["stage"] == 3:
        config["training"]["lr"] = 0.01


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Causal prefix training; missing pretrained/native prerequisites fail explicitly"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="explicit structural training: 64 crops, 2-frame loop, no native/perceptual teacher",
    )
    args = parser.parse_args(argv)
    config = copy.deepcopy(read_config(args.config))
    config["purpose"] = "train"
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.smoke:
        apply_smoke_config(config)
    if args.checkpoint:
        config["paths"]["checkpoint"] = args.checkpoint
    device = args.device or config["device"]
    try:
        preflight(config, "train", args.allow_untrained)
        seed_everything(config["seed"], device, config["gpu_memory_fraction"])
        model, initialization, artifact = build_model(config, device, args.allow_untrained, args.checkpoint)
        status = (
            "untrained_backbone_smoke"
            if args.smoke or initialization != "trained" and args.allow_untrained
            else "trained"
        )
        report = run_training(model, config, args.max_steps, artifact, args.resume, status)
        print(
            f"checkpoint={report['checkpoint']} status={report['status']} steps={report['completed_steps']}"
        )
    except MissingPrerequisites as error:
        report_missing(resolve(config["paths"]["output"]) / "train_prerequisites.json", error, config)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
