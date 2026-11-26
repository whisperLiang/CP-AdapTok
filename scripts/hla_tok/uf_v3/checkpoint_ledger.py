"""Verify V3 retained checkpoint counters against AdamW tensor states and run ledgers."""

import argparse
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.v2.protocol import json_at

GROUPS = ("native_i", "native_p", "quantizer", "refiner")


def inspect(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    optimizer = state["optimizer"]
    spec = dict(zip(GROUPS, optimizer["param_groups"], strict=True))
    counter = {}
    for group, layout in spec.items():
        operated = [int(optimizer["state"].get(key, {}).get("step", 0)) for key in layout["params"]]
        if len(operated) != len(state[group]):
            raise ValueError(f"{checkpoint.name} has a mismatched optimizer parameter group")
        counter[group] = {"tensors": len(operated), "adam_step_counts": sorted(set(operated)),
                          "elements": sum(param.numel() for param in state[group].values())}
    if max(counter["refiner"]["adam_step_counts"]) != state["step"]:
        raise ValueError("D1 AdamW counters differ from the saved global step")
    return {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size,
            "seed": state["signature"]["seed"], "saved_step": state["step"], "groups": counter}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--discarded-attempt", type=Path)
    args = parser.parse_args()
    runs = [json_at(path) for path in sorted((args.results / "training").glob("*/summary.json"))]
    outputs = [inspect(path) for path in sorted(args.checkpoints.glob("r0-*.pt"))]
    for run in runs:
        matches = [row for row in outputs if row["sha256"] == run["checkpoint_sha256"]]
        if len(matches) != 1 or matches[0]["saved_step"] != run["optimizer_updates"]:
            raise ValueError("logged updates do not match a retained checkpoint")
    discarded = inspect(args.discarded_attempt) if args.discarded_attempt else None
    write_json(args.results / "training/checkpoint_ledger.json", {
        "checkpoints": outputs, "completed_run_optimizer_calls": sum(run["added"] for run in runs),
        "discarded_reporting_attempt": discarded,
        "cuda_event_interval_seconds": sum(run["gpu_event_interval_seconds"] for run in runs),
        "wall_seconds": sum(run["wall_seconds"] for run in runs),
        "teacher_wall_seconds": sum(run["teacher_wall_seconds"] for run in runs),
        "frame_instances_with_replays": sum(run["frame_instances"] for run in runs),
    })


if __name__ == "__main__":
    main()
