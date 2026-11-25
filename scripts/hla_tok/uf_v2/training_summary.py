"""Verify actual saved optimizer nodes and collate experiment costs."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.v2.protocol import csv_at

NODES = {
    "seed1": [500, 2000], "seed2": [500, 2000],
    "joint-seed1": [4000, 6000, 8000], "joint-seed2": [4000, 6000, 8000],
    "formal-seed1": [10000, 12000, 14000, 16000, 18000, 20000],
    "formal-seed2": [10000, 12000, 14000, 16000, 18000, 20000],
}


def verify_resume_state(state: dict, row: dict, step: int) -> int:
    joint = "native_adjustments" in state
    proposed_position = state.get("next_data_position", state["step"])
    if state["signature"] != row["signature"] or proposed_position != step:
        raise ValueError("saved source, data signature or resume position differs")
    expected = step - 2000 if joint else step
    updates = max(int(value["step"]) for value in state["optimizer"]["state"].values())
    if updates != expected:
        raise ValueError("optimizer moments did not carry the full stage update count")
    keys = ("rng", "cuda_rng") if joint else ("cpu_rng_state", "cuda_rng_state")
    if any(state[key].dtype != torch.uint8 or state[key].numel() == 0 for key in keys):
        raise ValueError("checkpoint lacks a usable CPU or CUDA generator state")
    return updates


def record(folder: Path, path: Path) -> dict:
    row = json.loads(path.read_text())
    checkpoint = Path(row["checkpoint"])
    observed = sha256_file(checkpoint)
    if observed != row["checkpoint_sha256"] or row["read_counts"]["test"]:
        raise ValueError(f"training checkpoint identity or Test read failure: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    step = int(state["step"])
    logged = row.get("actual_optimizer_steps", row.get("optimizer_steps"))
    if step != logged:
        raise ValueError(f"checkpoint and optimizer step evidence differ: {checkpoint}")
    optimizer_steps = verify_resume_state(state, row, step)
    added = row.get("added", row.get("added_steps"))
    logs = csv_at(folder / f"steps-{step - added + 1}-{step}.csv")
    recorded = [int(r["step"]) for r in logs]
    if recorded != list(range(step - added + 1, step + 1)):
        raise ValueError("per-step training evidence is incomplete")
    result = {
        "stage": folder.name, "seed": row["seed"], "cumulative": step, "added": added,
        "unroll": row.get("unroll", (row.get("stages") or {}).get("unroll")), "gpu_hours": row["gpu_hours"],
        "frames": row["train_frame_instances"], "cache_seconds": row["cache_seconds"],
        "peak_bytes": row["peak_allocated_bytes"], "checkpoint": str(checkpoint), "sha256": observed,
        "max_parameter_updates_in_optimizer": optimizer_steps, "total_test_reads": row["read_counts"]["test"],
        "latest_gradient_norm": float(logs[-1]["gradient_norm"]), "mean_step_s": mean(float(r["step_s"]) for r in logs),
    }
    if "estimated_bpp" in logs[0]:
        result |= {"mean_rate_proxy_bpp": mean(float(r["estimated_bpp"]) for r in logs),
                   "mean_physical_cache_bpp": mean(float(r["actual_cache_bpp"]) for r in logs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results / "training"
    rows = []
    for name, targets in NODES.items():
        folder = root / name
        for step in targets:
            path = folder / f"step-{step}.json"
            if path.is_file():
                rows.append(record(folder, path))
    write_csv(root / "optimizer_ledger.csv", rows)
    write_json(root / "optimizer_ledger.json", {"nodes": rows, "optimizer_updates": {
        seed: sum(row["added"] for row in rows if row["seed"] == seed) for seed in (1, 2)}})


if __name__ == "__main__":
    main()
