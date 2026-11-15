"""Actual 1080p RD for the matched learned/spatial/grid controls."""

import argparse

import torch

from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.runtime import ROOT, build_model, read_config, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["learned_512", "spatial_512", "grid34_512"])
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    root = ROOT / "results/optimization_20261115/capacity"
    rows = []
    for name in args.variants:
        config = read_config(root / name / "config.json")
        config["evaluation"]["save_frames"] = False
        seed_everything(config["seed"], config["device"], 0.8)
        model, _, _ = build_model(
            config, config["device"], checkpoint_override=str(root / name / "latest.pt")
        )
        report = evaluate_sequences(
            model.eval(), config, output=root / name / "native1080", max_frames=args.frames, status="trained"
        )
        rows.extend([{"model": name, **r} for r in report["budget_summary"]])
        write_json(root.parent / "capacity_native1080.json", rows)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
