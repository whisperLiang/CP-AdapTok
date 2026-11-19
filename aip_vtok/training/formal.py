"""Fixed 1000-update dual-seed Gate 1 pilot; no sealed test access."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from aip_vtok.models.codec import LEGACY_MODEL_VERSION
from aip_vtok.runtime import file_hash, load_config, resolve, save_checkpoint, write_json
from aip_vtok.training.checkpoint import checkpoint_artifact, restore
from aip_vtok.training.components import TrainingComponents, create_components
from aip_vtok.training.data import prefetched_windows, sources, training_plans
from aip_vtok.training.diagnostics import gradient_diagnostics
from aip_vtok.training.paired_step import paired_step


def settings(path, seed):
    declared = yaml.safe_load(resolve(path).read_text())
    config = load_config(declared["gate0_config"])
    config["seed"], config["stage1"] = seed, declared["stage1"]
    torch.manual_seed(seed)
    return config


def require_prerequisites(config):
    directory = resolve(config["output"])
    for name in (
        "gate0_backbone_and_ref.json",
        "gate1_smoke/decision.json",
        "training_path_equivalence.json",
    ):
        if not json.loads((directory / name).read_text())["passed"]:
            raise RuntimeError(f"required prerequisite failed: {name}")


def training_input(config):
    rows, hashes = sources(config, "train")
    hashes.update(
        {
            "alignment": file_hash(resolve(config["checkpoints"]) / "alignment.pt"),
            "projector": file_hash(resolve(config["checkpoints"]) / "projector_w128/latest.pt"),
        }
    )
    return rows, hashes


@dataclass
class FormalRun:
    config: dict
    components: TrainingComponents
    directory: Path
    checkpoint: Path
    rows: list
    hashes: dict
    plans: list
    start: int
    history: list


def prepare(config, resume):
    require_prerequisites(config)
    label = f"gate1_seed{config['seed']}"
    directory = resolve(config["output"]) / label
    checkpoint = resolve(config["checkpoints"]) / label / "latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint}: use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    components = create_components(config, model_version=LEGACY_MODEL_VERSION)
    rows, hashes = training_input(config)
    plans = training_plans(
        rows, config["stage1"]["steps"], config["stage1"]["sequence_length"], components.rng
    )
    start, history = 0, []
    if resume:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start, history = restore(
            artifact, components.codec, components.optimizer, components.rng, config, hashes
        )
        plans = artifact["plans"]
    write_json(
        directory / "inputs.json",
        {
            "config": config,
            "hashes": hashes,
            "plans": plans,
            "sealed_test_opened": False,
            "initialization": "fresh seed; smoke weights are not promoted",
        },
    )
    return FormalRun(config, components, directory, checkpoint, rows, hashes, plans, start, history)


def update(run, step, clip):
    config, components = run.config, run.components
    stage = config["stage1"]
    if run.plans[step - 1]["flip"]:
        clip = clip.flip(-1)
    components.optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=stage["amp_bf16"]):
        loss, records = paired_step(
            components.codec,
            clip,
            components.objective,
            components.rng,
            step % stage.get("gradient_every", 100) == 0 or step == 1,
            weights=stage.get("alignment_loss"),
            perceptual_frames=stage.get("perceptual_frames"),
        )
    gradients = gradient_diagnostics(components.codec)
    norm = torch.nn.utils.clip_grad_norm_(components.parameters, 1.0, error_if_nonfinite=True)
    components.optimizer.step()
    return {
        "step": step,
        "loss": loss,
        "records": records,
        "module_gradients": gradients,
        "grad_norm": float(norm),
        "rate_weight": 0.0,
        "source": run.rows[run.plans[step - 1]["source"]]["content_id"],
        "source_start": run.plans[step - 1]["start"],
    }


def record_progress(run, step, started):
    config, c = run.config, run.components
    if step % 10 == 0 or step == 1:
        print(
            f"seed={config['seed']} step={step}/{config['stage1']['steps']} "
            f"loss={run.history[-1]['loss']:.6f} elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
    if step % config["stage1"]["save_every"] == 0 or step == config["stage1"]["steps"]:
        artifact = checkpoint_artifact(step, c.codec, c.optimizer, c.rng, run.history, config, run.hashes)
        artifact["plans"] = run.plans
        progress = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "wall_s_current_session": time.perf_counter() - started,
        }
        if "model_version" in config:
            artifact["progress"] = progress
        save_checkpoint(run.checkpoint, artifact)
        publish_snapshot(run.checkpoint, step, config["stage1"].get("snapshots", ()))
        write_json(run.directory / "training.json", {"history": run.history, "steps": step, **progress})


def publish_snapshot(checkpoint, step, snapshots):
    if step not in snapshots:
        return
    snapshot = checkpoint.with_name(f"step_{step:04d}.pt")
    if not snapshot.exists() or file_hash(checkpoint) != file_hash(snapshot):
        temporary = snapshot.with_suffix(".link")
        temporary.unlink(missing_ok=True)
        temporary.hardlink_to(checkpoint)
        temporary.replace(snapshot)


def finish(run):
    if len(run.history) != run.config["stage1"]["steps"]:
        raise RuntimeError("training history is incomplete")
    write_json(
        run.directory / "training_complete.json",
        {
            "steps": run.config["stage1"]["steps"],
            "checkpoint_sha256": file_hash(run.checkpoint),
            "seed": run.config["seed"],
            "hashes": run.hashes,
        },
    )


def train(config, resume=False):
    run = prepare(config, resume)
    return run_updates(run, prefetched_windows(run.rows, run.plans[run.start :]))


def run_updates(run, clips):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, clip in enumerate(clips, run.start + 1):
        run.history.append(update(run, step, clip))
        record_progress(run, step, started)
    finish(run)
    return run.components.codec.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aip_vtok/gate1.yaml")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(settings(args.config, args.seed), args.resume)


if __name__ == "__main__":
    main()
