"""Execute bounded native-pixel curriculum runs; retain all configs and recoverable artifacts."""

import argparse
import json

import torch

from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.runtime import build_model, load_checkpoint, read_config, resolve, seed_everything, write_json
from cp_adaptok.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["d1", "d2", "d3", "d4"])
    parser.add_argument("--steps", nargs="+", type=int, default=[300, 100, 100, 100])
    parser.add_argument("--initial", default="results/optimization_20261115/ablations/balanced_vq/latest.pt")
    args = parser.parse_args()
    if len(args.steps) != len(args.stages):
        parser.error("supply one update budget per stage")
    manifest = json.loads(resolve("data/davis2017/manifest.json").read_text())
    names = {"d1": "d1_vq_512", "d2": "d2_causal_720", "d3": "d3_causal_1080", "d4": "d4_long_1080"}
    for stage, steps in zip(args.stages, args.steps):
        config = read_config(f"configs/{names[stage]}.yaml")
        config["purpose"] = "train"
        config["data"]["train"] = manifest["training_specs"]
        config["training"]["query_initialization_scale"] = 1.0
        config["training"]["grid_query_initialization_scale"] = 1.0
        config["training"]["vq"].update(
            warmup_steps=0, initialization=None, temperature=0.01, final_temperature=0.002, anneal_steps=steps
        )
        if stage == "d1":
            config["paths"]["checkpoint"] = args.initial
        checkpoint = resolve(config["paths"]["save_checkpoint"])
        resume = checkpoint.is_file()
        artifact = load_checkpoint(checkpoint) if resume else None
        output = resolve(config["paths"]["output"])
        write_json(output / "config.json", config)
        seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
        model, _, _ = build_model(
            config, config["device"], checkpoint_override=str(checkpoint) if resume else None
        )
        report = run_training(model, config, steps, artifact, resume, "trained")
        model.eval()
        model.refresh_identity()
        evaluate_sequences(
            model,
            config,
            budgets=config["evaluation"]["budgets"],
            status="trained",
            output=output / "test_native1080",
            max_frames=8,
        )
        print(
            json.dumps({"stage": stage, "updates": report["completed_steps"], "converged": False}), flush=True
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
