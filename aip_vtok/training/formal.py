"""Fixed 1000-update dual-seed Gate 1 pilot; no sealed test access."""

import argparse
import json
import time

import torch
import yaml

from aip_vtok.models.codec import create_codec
from aip_vtok.runtime import file_hash, load_config, resolve, save_checkpoint, write_json
from aip_vtok.training.data import prefetched_windows, sources, training_plans
from aip_vtok.training.losses import ReconstructionLoss
from aip_vtok.training.paired_step import paired_step
from aip_vtok.training.smoke import checkpoint_artifact, gradient_diagnostics, restore
from aip_vtok.training.trainer import RNGStreams


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


def train(config, resume=False):
    require_prerequisites(config)
    label = f"gate1_seed{config['seed']}"
    directory = resolve(config["output"]) / label
    checkpoint = resolve(config["checkpoints"]) / label / "latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint}: use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    codec = create_codec(config).train()
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    parameters = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["stage1"]["lr"], weight_decay=0)
    rng = RNGStreams(config["seed"])
    rows, hashes = training_input(config)
    plans = training_plans(rows, config["stage1"]["steps"], config["stage1"]["sequence_length"], rng)
    start, history = 0, []
    if resume:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start, history = restore(artifact, codec, optimizer, rng, config, hashes)
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
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, clip in enumerate(prefetched_windows(rows, plans[start:]), start + 1):
        if plans[step - 1]["flip"]:
            clip = clip.flip(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["stage1"]["amp_bf16"]):
            loss, records = paired_step(codec, clip, objective, rng, step % 100 == 0 or step == 1)
        gradients = gradient_diagnostics(codec)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": loss,
                "records": records,
                "module_gradients": gradients,
                "grad_norm": float(norm),
                "rate_weight": 0.0,
                "source": rows[plans[step - 1]["source"]]["content_id"],
                "source_start": plans[step - 1]["start"],
            }
        )
        if step % 10 == 0 or step == 1:
            print(
                f"seed={config['seed']} step={step}/{config['stage1']['steps']} "
                f"loss={loss:.6f} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if step % config["stage1"]["save_every"] == 0:
            artifact = checkpoint_artifact(step, codec, optimizer, rng, history, config, hashes)
            artifact["plans"] = plans
            save_checkpoint(checkpoint, artifact)
            write_json(
                directory / "training.json",
                {
                    "history": history,
                    "steps": step,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "wall_s_current_session": time.perf_counter() - started,
                },
            )
    write_json(
        directory / "training_complete.json",
        {
            "steps": config["stage1"]["steps"],
            "checkpoint_sha256": file_hash(checkpoint),
            "seed": config["seed"],
            "hashes": hashes,
        },
    )
    return codec.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aip_vtok/gate1.yaml")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(settings(args.config, args.seed), args.resume)


if __name__ == "__main__":
    main()
