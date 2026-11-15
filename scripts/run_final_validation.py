"""Calibrate on training tokens and evaluate selected real bitstream candidates independently."""

import argparse
import copy

from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    config = read_config("results/optimization_20261115/multiscene_extended/config.json")
    config["paths"].update(checkpoint="results/optimization_20261115/multiscene_extended/latest.pt")
    config["purpose"] = "train"
    config["evaluation"]["save_frames"] = False
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"])
    model.eval()
    evaluate_sequences(
        model,
        config,
        split="validation",
        output="results/optimization_20261115/extended_validation",
        max_frames=8,
    )
    config["stage"] = 3
    config["teacher"] = {"policy": "none"}
    config["loss"].update(latent=0.0, lpips=0.0, dists=0.0, vq=0.0)
    config["training"].update(
        lr=0.01,
        sequence_length=4,
        save_every=100,
        query_initialization_scale=1.0,
        grid_query_initialization_scale=1.0,
    )
    config["paths"].update(
        output="results/optimization_20261115/auxiliary",
        save_checkpoint="results/optimization_20261115/auxiliary/latest.pt",
    )
    write_json("results/optimization_20261115/final_config.json", config)
    run_training(model, copy.deepcopy(config), args.steps, status="trained")
    model.eval()
    model.refresh_identity()
    evaluate_sequences(
        model, config, output="results/optimization_20261115/final_evaluation/eval", max_frames=8
    )
    evaluate_sequences(
        model,
        config,
        output="results/optimization_20261115/final_long/eval",
        metric_names=["psnr"],
        max_frames=32,
    )
    write_json(
        "results/optimization_20261115/selection.json",
        {
            "selected_checkpoint": "results/optimization_20261115/auxiliary/latest.pt",
            "basis": "extended representation plus new deterministic-token entropy/scorer calibration",
            "scientific_quality_gate_passed": False,
            "formal_convergence_claim": False,
            "old_scorer_or_entropy_used_for_final_rd": False,
        },
    )


if __name__ == "__main__":
    main()
