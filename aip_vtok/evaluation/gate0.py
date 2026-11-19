"""Gate 0 entrypoint: run real baselines, fit, then evaluate before releasing work."""

import argparse

from aip_vtok.evaluation.evaluate import create_backbone, evaluate_projector, load_alignment, run_baseline
from aip_vtok.runtime import load_config
from aip_vtok.training.trainer import fit_projector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "train"))
    parser.add_argument("--config", default="configs/aip_vtok/gate0.yaml")
    parser.add_argument("--width", type=int, choices=(128, 256), default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "baseline":
        run_baseline(config)
    else:
        backbone = create_backbone(config)
        load_alignment(config, backbone)
        projector = fit_projector(config, backbone, args.width, args.resume)
        result = evaluate_projector(config, backbone, projector)
        print(result["summary"], result["projector_latency"], flush=True)


if __name__ == "__main__":
    main()
