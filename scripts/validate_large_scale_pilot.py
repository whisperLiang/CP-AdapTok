"""Bounded real rANS / native1080p pilot validation; never opens the sealed final test."""

import argparse

import torch

from cp_adaptok.large_scale import gpu_snapshot, memory_fraction, read_experiment
from cp_adaptok.large_scale_validation import validate_checkpoint
from cp_adaptok.runtime import build_model, read_config, resolve, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pilot_20261116")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--sequences", type=int, default=2)
    args = parser.parse_args()
    manifest, splits = read_experiment(args.run_id)
    output = resolve("results/large_scale") / manifest["run_id"] / "pilot" / "shared"
    config = read_config(output / "config.json")
    snapshot = gpu_snapshot()
    memory_fraction(snapshot, config["large_scale"])
    checkpoint = resolve(config["paths"]["save_checkpoint"]).with_name("best_validation.pt")
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=str(checkpoint))
    try:
        report = validate_checkpoint(model, config, splits, "full", output / "pilot_real_rd", checkpoint,
                                     settings_override={"sequences": args.sequences, "frames": args.frames})
        report.update(pilot_only=True, sufficient_training=False, final_test_used=False,
                      scorer_status="not retrained for pilot VQ; prediction is diagnostic only")
        write_json(output / "pilot_rd_summary.json", {k: report[k] for k in
            ("budget_summary", "budget_summary_by_frame_type", "quality_drop_ratio", "prefix_gains",
             "entropy_id", "model_id", "pilot_only", "sufficient_training", "final_test_used", "scorer_status")})
        print(f"pilot_real_rans_frames={report['sample_count']} model_id={report['model_id']}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
