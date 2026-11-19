"""Replay one real native-1080p update twice without changing the saved model."""

import copy
import gc
import os

import torch

from aip_vtok.models.codec import LEGACY_MODEL_VERSION
from aip_vtok.runtime import file_hash, resolve, write_json
from aip_vtok.training.checkpoint import restore, trainable_state
from aip_vtok.training.components import create_components
from aip_vtok.training.data import read_window, training_plans
from aip_vtok.training.formal import settings, training_input
from aip_vtok.training.paired_step import paired_step


def replay(config, artifact):
    artifact = copy.deepcopy(artifact)  # Optimizer step tensors must not alias another replay.
    c = create_components(config, model_version=LEGACY_MODEL_VERSION)
    rows, hashes = training_input(config)
    restore(artifact, c.codec, c.optimizer, c.rng, config, hashes)
    plan = training_plans(rows, 1, 4, c.rng)[0]
    clip = read_window(rows[plan["source"]], plan["start"], 4)
    if plan["flip"]:
        clip = clip.flip(-1)
    c.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, records = paired_step(c.codec, clip, c.objective, c.rng)
    torch.nn.utils.clip_grad_norm_(c.parameters, 1.0, error_if_nonfinite=True)
    c.optimizer.step()
    torch.cuda.synchronize()
    return {
        "loss": loss,
        "plan": plan,
        "records": records,
        "parameters": {k: v.detach().cpu().clone() for k, v in trainable_state(c.codec).items()},
        "rng": c.rng.state_dict(),
    }


def main():
    config = settings("configs/aip_vtok/gate1.yaml", 1)
    path = resolve(config["checkpoints"]) / "gate1_seed1/latest.pt"
    before = file_hash(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    first = replay(config, artifact)
    gc.collect()
    torch.cuda.empty_cache()
    second = replay(config, artifact)
    errors = {
        name: (first["parameters"][name] - value).abs().max().item()
        for name, value in second["parameters"].items()
    }
    result = {
        "source_step": artifact["step"],
        "maximum_parameter_error": max(errors.values()),
        "loss_error": abs(first["loss"] - second["loss"]),
        "sample_plan_equal": first["plan"] == second["plan"],
        "four_rng_states_equal": all(
            torch.equal(first["rng"][name], second["rng"][name]) for name in first["rng"]
        ),
        "checkpoint_unchanged": before == file_hash(path),
        "scope": "two temporary continuation updates; no saved-model changes",
    }
    result["runtime"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    result["passed"] = (
        result["maximum_parameter_error"] == 0
        and result["loss_error"] == 0
        and result["sample_plan_equal"]
        and result["four_rng_states_equal"]
        and result["checkpoint_unchanged"]
    )
    write_json("results/aip_vtok/checkpoint_gpu_replay.json", result)
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
