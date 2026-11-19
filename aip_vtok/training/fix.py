"""Train isolated Gate 1-Fix stages with reproducible checkpoints."""

import argparse
import json
from itertools import repeat

import torch
import yaml

from aip_vtok.models.codec import MODEL_VERSION
from aip_vtok.runtime import file_hash, resolve, write_json
from aip_vtok.training.checkpoint import load_fix_weights, require_identity, restore, restore_training_state
from aip_vtok.training.components import create_components, enable_projector
from aip_vtok.training.data import prefetched_windows, read_window, source_by_id, training_plans
from aip_vtok.training.formal import FormalRun, publish_snapshot, run_updates, settings, training_input

CONFIG = "configs/aip_vtok/gate1_fix.yaml"
RESULTS = resolve("results/aip_vtok/gate1_fix")
CHECKPOINTS = resolve("checkpoints/aip_vtok/gate1_fix")
PHASES = ("smoke", "fix_a", "b0", "b1", "fix_c")


def read_json(path):
    return json.loads(path.read_text())


def directories(phase, seed):
    if phase == "smoke":
        label = "fix_a/smoke"
    elif phase in ("b0", "b1"):
        label = f"fix_b/{phase}"
    else:
        label = f"{phase}/seed{seed}"
    return RESULTS / label, CHECKPOINTS / label


def fix_settings(phase, seed, path=CONFIG, *, joint_projector=False):
    declared = yaml.safe_load(resolve(path).read_text())
    if declared["model_version"] != MODEL_VERSION or declared["use_temporal_predictor"] is not False:
        raise ValueError("Gate 1-Fix requires the predictor-free model version")
    if phase != "fix_c" and seed != 1:
        raise ValueError("Fix-A and the matched Fix-B branches use seed 1")
    base = settings(declared["gate1_config"], seed)
    phase_config = declared["fix_b" if phase in ("b0", "b1") else phase]
    base.update(
        model_version=declared["model_version"],
        use_temporal_predictor=False,
        fix_phase=phase,
        fix_limits=declared["limits"],
        joint_projector=phase == "b1" or (phase == "fix_c" and joint_projector),
    )
    base["stage1"].update(phase_config, alignment_loss=declared["alignment_loss"])
    if phase == "fix_c" and (
        phase_config["rollout"] != "rollout_8f_detached_ref" or phase_config["sequence_length"] != 8
    ):
        raise ValueError("Fix-C requires eight consecutive frames and detached reference commits")
    if phase == "fix_c" and not 4 <= phase_config["perceptual_frames"] <= 8:
        raise ValueError("Fix-C requires LPIPS on at least the last four frames")
    return base


def require_release(phase):
    gates = {
        "fix_a": "fix_a/smoke/decision.json",
        "b0": "fix_a/decision.json",
        "b1": "fix_a/decision.json",
        "fix_c": "fix_b/decision.json",
    }
    if phase == "fix_c":
        require_release("b0")
    if phase in gates:
        path = RESULTS / gates[phase]
        if not path.is_file() or not read_json(path)["passed"]:
            raise RuntimeError(f"phase {phase} requires a passing {path}")
        return read_json(path)
    return {}


def fork_parent(config, path):
    step = config["stage1"]["fork_step"]
    parent = directories("fix_a", 1)[1] / f"step_{step:04d}.pt"
    artifact = torch.load(parent, map_location="cpu", weights_only=False)
    expected = fix_settings("fix_a", 1, path)
    _, hashes = training_input(expected)
    require_identity(artifact, expected, hashes)
    if artifact["step"] != step or step + config["stage1"]["steps"] > len(artifact["plans"]):
        raise ValueError("Fix-B needs the declared Fix-A step and the full matched data/augmentation plan")
    return artifact, file_hash(parent)


def construct_run(config, config_path, resume):
    phase, seed = config["fix_phase"], config["seed"]
    directory, weights = directories(phase, seed)
    checkpoint = weights / "latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint}: use --resume")
    rows, hashes = training_input(config)
    stage = config["stage1"]
    parent = None
    if phase in ("b0", "b1"):
        parent, hashes["fork_checkpoint"] = fork_parent(config, config_path)
    initializing_fork = parent is not None and not resume
    components = create_components(
        config,
        model_version=config["model_version"],
        joint_projector=config["joint_projector"] and not initializing_fork,
    )
    start, history = 0, []
    if resume:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start, history = restore(
            artifact, components.codec, components.optimizer, components.rng, config, hashes
        )
        plans = artifact["plans"]
        publish_snapshot(checkpoint, start, stage.get("snapshots", ()))
        if "progress" in artifact:
            write_json(
                directory / "training.json", {"history": history, "steps": start, **artifact["progress"]}
            )
    elif initializing_fork:
        load_fix_weights(components.codec, parent)
        restore_training_state(parent, components.optimizer, components.rng)
        if config["joint_projector"]:
            enable_projector(components.codec, components.optimizer, components.parameters)
        plans = parent["plans"][stage["fork_step"] : stage["fork_step"] + stage["steps"]]
    elif phase == "smoke":
        row = source_by_id(rows, stage["video"])
        plans = [
            {
                "source": rows.index(row),
                "start": stage["start"],
                "length": stage["sequence_length"],
                "flip": False,
            }
        ] * stage["steps"]
    else:
        plans = training_plans(rows, stage["steps"], stage["sequence_length"], components.rng)
    write_json(directory / "inputs.json", {"config": config, "hashes": hashes, "plans": plans})
    return FormalRun(config, components, directory, checkpoint, rows, hashes, plans, start, history)


def train_fix(phase, seed, config_path=CONFIG, *, resume=False):
    release = require_release(phase)
    config = fix_settings(phase, seed, config_path, joint_projector=release.get("winner") == "b1")
    gate0 = read_json(resolve(config["output"]) / "gate0_backbone_and_ref.json")
    manifest = read_json(resolve(config["output"]) / "checkpoints_manifest.json")
    projector = resolve(config["checkpoints"]) / "projector_w128/latest.pt"
    if not gate0["passed"] or file_hash(projector) != manifest["projector_w128/latest.pt"]["sha256"]:
        raise RuntimeError("Gate 1-Fix requires the unchanged Gate 0 Projector")
    run = construct_run(config, config_path, resume)
    remaining = run.plans[run.start :]
    if phase == "smoke":
        plan = run.plans[0]
        clip = read_window(run.rows[plan["source"]], plan["start"], plan["length"])
        return run_updates(run, repeat(clip, len(remaining)))
    return run_updates(run, prefetched_windows(run.rows, remaining))


def phase_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    return parser


def main():
    parser = phase_parser(__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_fix(args.phase, args.seed, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
