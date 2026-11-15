"""Explicitly migrated shared aggregation versus two matched global-softmax controls."""

import argparse

from scripts.compare_prefix_training import matched_config, run_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=["C0", "C1", "C2"], default=["C0", "C1", "C2"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--root", default="results/prefix_utilization/decoder_ablations")
    parser.add_argument("--checkpoint", default="results/optimization_20261115/auxiliary/latest.pt")
    args = parser.parse_args()
    for name in args.variants:
        config = matched_config(name, args.root, args.steps, args.crop, args.checkpoint)
        if name == "C2":
            config["model"]["token_aggregation"] = "shared_progressive_v2"
            config["paths"]["checkpoint"] = None
        run_variant(name, config, args.steps, migration_source=args.checkpoint if name == "C2" else None)


if __name__ == "__main__":
    main()
