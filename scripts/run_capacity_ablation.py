"""Matched short native-pixel capacity controls; no inheritance of mismatched tensors."""

import argparse
import json

import torch

from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["learned", "spatial", "grid34", "n4096", "n8192", "c1024", "c2048", "c4096"],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--crop", type=int, default=256)
    args = parser.parse_args()
    for name in args.variants:
        config = read_config("configs/d1_vq_512.yaml")
        config["purpose"] = "train"
        config["paths"].update(
            checkpoint=None,
            output=f"results/optimization_20261115/capacity/{name}_{args.crop}",
            save_checkpoint=f"results/optimization_20261115/capacity/{name}_{args.crop}/latest.pt",
        )
        config["training"].update(
            crop=args.crop, fixed_sample_seed=19, lr=0.0003, all_budgets=True, save_every=args.steps
        )
        config["training"]["vq"].update(
            warmup_steps=args.steps // 2, temperature=0.1, final_temperature=0.01, anneal_steps=args.steps
        )
        config["teacher"] = {"policy": "i_only"}
        if name == "spatial":
            config["control"] = "fixed_spatial"
            config["training"].pop("query_initialization_scale")
        if name == "grid34":
            config["model"]["grid"] = [34, 60]
        if name.startswith("n"):
            config["model"]["n_max"] = int(name[1:])
            config["training"]["budgets"] = [128, 512, 2048, int(name[1:])]
        if name.startswith("c"):
            config["model"]["codebook_size"] = int(name[1:])
        write_json(config["paths"]["output"] + "/config.json", config)
        seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
        model, _, _ = build_model(config, config["device"])
        run_training(model, config, args.steps, status="trained")
        model.refresh_identity()
        # Fixed crop diagnostics use the exact same source window as the overfit run.
        from scripts.run_optimization_experiments import diagnostic_reconstruction

        result = diagnostic_reconstruction(model, config)
        write_json(config["paths"]["output"] + "/diagnostic_evaluation.json", result)
        print(json.dumps({"variant": name, "updates": args.steps, "converged": False}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
