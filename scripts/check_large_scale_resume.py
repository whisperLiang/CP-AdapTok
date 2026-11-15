"""Verify native CUDA checkpoint/sampling restoration, reporting arithmetic nondeterminism explicitly."""

import argparse
import copy
import json

import torch

from cp_adaptok.runtime import build_model, load_checkpoint, resolve, seed_everything, write_json
from cp_adaptok.training import run_training


def advance(source, output, target):
    initial = load_checkpoint(source)
    config = copy.deepcopy(initial["config"])
    config["paths"].update(checkpoint=str(source), output=str(output), save_checkpoint=str(output / "latest.pt"))
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, artifact = build_model(config, config["device"], checkpoint_override=str(source))
    try:
        run_training(model, config, target, artifact, resume=True, status="trained")
    finally:
        del model
        torch.cuda.empty_cache()
    return output / "latest.pt"


def tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def tree_close(left, right):
    if torch.is_tensor(left):
        return torch.allclose(left, right, atol=5e-5, rtol=1e-4) if left.is_floating_point() else torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_close(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(tree_close(a, b) for a, b in zip(left, right))
    return left == right


def sampled_inputs(path):
    result = []
    for line in (path / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        result.append((row["step"], [(r["source"], r["source_frame"], r["augmentation"],
                                     [b["tokens"] for b in r["budgets"]]) for r in row["records"]]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/large_scale/pilot_20261116/pilot/shared/best_validation.pt")
    parser.add_argument("--output", default="results/large_scale/native_resume_numeric")
    args = parser.parse_args()
    source, output = resolve(args.checkpoint), resolve(args.output)
    start = load_checkpoint(source)["step"]
    whole = advance(source, output / "whole", start + 2)
    first = advance(source, output / "split", start + 1)
    second = advance(first, output / "split", start + 2)
    a, b = load_checkpoint(whole), load_checkpoint(second)
    checks = {key: tree_equal(a[key], b[key]) for key in ("model", "optimizer", "scheduler", "rng_streams",
                                                        "torch_rng", "cuda_rng", "vq_training_state")}
    numeric = {key: tree_close(a[key], b[key]) for key in ("model", "optimizer")}
    same_inputs = sampled_inputs(output / "whole") == sampled_inputs(output / "split")
    rng_exact = all(value for key, value in checks.items() if key not in ("model", "optimizer"))
    differences = [float((a["model"][k] - b["model"][k]).abs().max()) for k in a["model"]
                   if a["model"][k].is_floating_point() and a["model"][k].numel()]
    result = {"source_checkpoint": str(source), "source_step": start, "target_step": start + 2,
              "real_optimizer_updates_executed": 4, "bitwise_checks": checks, "numeric_checks": numeric,
              "data_augmentation_and_budget_replay_exact": same_inputs, "max_model_absolute_difference": max(differences),
              "arithmetic_tolerance": {"atol": 5e-5, "rtol": 1e-4},
              "passed": rng_exact and same_inputs and all(numeric.values()),
              "cuda_bitwise_update_guarantee": False,
              "known_limit": "adaptive_avg_pool2d_backward_cuda has no deterministic implementation in installed torch",
              "purpose": "native512 pretrained stochastic-VQ resume acceptance; excluded from formal budget"}
    write_json(output / "acceptance.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
