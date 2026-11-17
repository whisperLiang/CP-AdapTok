"""Independent acceptance audit of immutable inputs and recorded real updates."""

import argparse
import json
import subprocess

import torch

from cp_adaptok.runtime import resolve, write_json
from experiments.fixed2048.checkpoint import sha256

RUNS = {
    "overfit1": ("continuous/overfit1", "overfit/1frame"),
    "overfit4": ("continuous/overfit4", "overfit/4frame"),
    "overfit16": ("continuous/overfit16", "overfit/16frame"),
    "R1": ("continuous/main", "continuous"),
    "R2": ("vq/main", "quantized/training"),
    "R3": ("grid34/main", "grid_ablation/decoder34"),
    "EncoderKV34": ("encoder34/main", "grid_ablation/encoder34"),
    "raw192": ("raw192/main", "raw192"),
}


def read(path):
    return json.loads(path.read_text())


def audit_update(checkpoint_directory, results_directory):
    initial = torch.load(checkpoint_directory / "step0.pt", map_location="cpu", weights_only=True)
    trained = torch.load(checkpoint_directory / "latest.pt", map_location="cpu", weights_only=True)
    frozen = [name for name in initial["model"] if name.startswith(("intra.", "ld."))]
    changed = [name for name in frozen if not torch.equal(initial["model"][name], trained["model"][name])]
    completion = read(results_directory / "completion.json")
    history = read(results_directory / "history.json")
    steps = [row["step"] for row in history["train"]]
    actual = completion["actual_optimizer_steps"]
    return {
        "steps": actual,
        "strictly_contiguous_steps": steps == list(range(1, actual + 1)),
        "checkpoint_step_matches": trained["step"] == actual,
        "all_frozen_backbone_tensors_equal": not changed,
        "changed_backbone_tensors": changed,
        "all_records_have_no_backbone_gradients": all(
            r["frozen_backbone_gradients"] == 0 for r in history["train"]
        ),
        "codebook_updated": not torch.equal(
            initial["model"]["vq.codebook.weight"], trained["model"]["vq.codebook.weight"]
        ),
        "completed_frames": completion["actual_training_frames"],
        "optimizer_wall_s": completion["optimizer_wall_s"],
    }


def audit_outputs(root):
    records = 0
    wrong_shapes = []
    for path in sorted(root.rglob("frame_metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            records += 1
            if row.get("rgb_shape") != [1, 3, 1080, 1920]:
                wrong_shapes.append({"file": str(path), "frame": row["frame"]})
            if row["real_bitstream"] is not False or row["bpp"] is not None:
                raise ValueError("reconstruction metrics falsely labelled as a real bitstream")
    return {"native_frame_records": records, "wrong_shapes": wrong_shapes, "all_native_rgb": not wrong_shapes}


def valid_update(name, row):
    required = (
        "strictly_contiguous_steps",
        "checkpoint_step_matches",
        "all_frozen_backbone_tensors_equal",
        "all_records_have_no_backbone_gradients",
    )
    return all(row[field] for field in required) and row["codebook_updated"] == (name == "R2")


def brief(result):
    updates = result["training_runs"]
    return {
        "actual_optimizer_updates": sum(r["steps"] for r in updates.values()),
        "actual_training_frames": sum(r["completed_frames"] for r in updates.values()),
        "outputs": result["outputs"],
        "production_diff_files": result["production_diff_files"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/fixed2048")
    args = parser.parse_args()
    root = resolve(args.root)
    updates = {
        name: audit_update(resolve("checkpoints/fixed2048") / checkpoints, root / results)
        for name, (checkpoints, results) in RUNS.items()
    }
    protected = subprocess.check_output(
        ["git", "diff", "origin/main", "--name-only", "--", "cp_adaptok", "scripts", "data/manifests"],
        text=True,
    ).splitlines()
    source = read(root / "source_identity.json")
    source_preserved = sha256(source["path"]) == source["sha256"]
    result = {
        "training_runs": updates,
        "outputs": audit_outputs(root),
        "production_diff_files": protected,
        "source_B1_checkpoint_preserved": source_preserved,
        "final_test_seal": read(root / "final_test/seal.json"),
        "mutation_results": read(root / "mutation_check.json"),
    }
    write_json(root / "acceptance.json", result)
    failures = [name for name, row in updates.items() if not valid_update(name, row)]
    if failures or protected or not source_preserved or not result["outputs"]["all_native_rgb"]:
        raise RuntimeError(f"acceptance failed: {failures}; protected={protected}")
    print(json.dumps(brief(result), indent=2))


if __name__ == "__main__":
    main()
