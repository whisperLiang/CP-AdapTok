"""Comparable, resumable short-run diagnostics. Never labels them converged."""

import argparse
import copy
import json
from pathlib import Path

import torch

from cp_adaptok.runtime import build_model, load_checkpoint, read_config, resolve, seed_everything, write_json
from cp_adaptok.training import run_training


def experiment_config(name, root, steps):
    config = read_config("configs/diagnostic_continuous.yaml")
    config["purpose"] = "train"
    train = config["training"]
    train.update(sequence_length=4, tbptt=2, lr=0.0003, save_every=steps)
    config["gpu_memory_fraction"] = 0.7
    config["paths"].update(
        checkpoint=None, save_checkpoint=str(root / name / "latest.pt"), output=str(root / name)
    )
    vq = train["vq"]
    if name.startswith("continuous"):
        train["sequence_length"] = int(name.split("_")[-1])
    else:
        vq["strategy"] = "deterministic_nearest" if name == "hard" else "stochastic_training"
    if name in ("warmup", "initialized", "dead"):
        vq["warmup_steps"] = steps // 2
    if name in ("initialized", "dead"):
        vq["initialization"] = {"clips": 8, "max_samples": 32768, "iterations": 10}
    if name == "annealed":
        vq.update(temperature=1.0, final_temperature=0.05, anneal_steps=steps)
    if name.startswith("balanced_") or name.startswith("content_"):
        train["query_initialization_scale"] = 50.0
        vq.update(
            strategy="stochastic_training",
            temperature=0.1,
            final_temperature=0.01,
            anneal_steps=steps,
            warmup_steps=steps // 2,
            initialization={"clips": 8, "max_samples": 32768, "iterations": 10},
        )
        if name in ("balanced_continuous", "content_continuous"):
            vq.clear()
            vq["strategy"] = "continuous"
        if name == "balanced_decoder":
            train["grid_query_initialization_scale"] = 50.0
    if name.startswith("content_"):
        config["model"]["query_content_only"] = True
        train["grid_query_initialization_scale"] = 50.0
    if name == "dead":
        vq["dead_codes"] = {"window": 20, "max_replacements": 16, "min_count": 1}
    if name.startswith("teacher_"):
        train["query_initialization_scale"] = 50.0
        vq.update(
            strategy="stochastic_training",
            warmup_steps=steps // 2,
            initialization={"clips": 8, "max_samples": 32768, "iterations": 10},
            temperature=0.1,
            final_temperature=0.01,
            anneal_steps=steps,
        )
        policy = {"teacher_T1": "legacy_ip", "teacher_T2": "i_only", "teacher_T3": "none"}[name]
        config["teacher"] = {"policy": policy, "allow_mismatched_p_diagnostic": policy == "legacy_ip"}
    return config


@torch.no_grad()
def diagnostic_reconstruction(model, config):
    from cp_adaptok.data import read_video, sample_training_spec
    from cp_adaptok.diagnostics import vq_diagnostics
    from cp_adaptok.metrics import MetricSuite
    from cp_adaptok.streaming.sender import ids_digest

    rng = torch.Generator().manual_seed(config["training"]["fixed_sample_seed"])
    spec = sample_training_spec(config["data"]["train"][0], config["training"]["sequence_length"], rng)
    clip = read_video(spec, config["training"]["crop"], rng)
    continuous = config["training"]["vq"]["strategy"] == "continuous"
    model.train(continuous)
    metrics = MetricSuite(model.device, ["psnr", "lpips", "dists"])
    state, rows = model.empty_state(), []
    for t, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode_full(rgb, state)
        row = {
            "frame": t,
            "frame_type": encoded.context.frame_type,
            "vq": vq_diagnostics(model.vq, encoded.features, encoded.vq),
            "prefixes": [],
        }
        for k in config["training"]["budgets"]:
            rec, _ = model.decode_embeddings(encoded.vq.embeddings[:, :k], encoded.context)
            row["prefixes"].append(dict(tokens=k, **metrics(rgb, rec)))
        _, feature = model.decode_embeddings(encoded.vq.embeddings[:, : model.config.k_ref], encoded.context)
        digest = ids_digest(encoded.prefix(model.config.k_ref)) if not continuous else "continuous-diagnostic"
        state = encoded.context.commit(feature, digest).detach()
        rows.append(row)
    return {
        "records": rows,
        "continuous_only_diagnostic": continuous,
        "real_bitstream_rd": False,
        "metric_errors": metrics.errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--root", type=Path, default=Path("results/optimization_20261115/ablations"))
    args = parser.parse_args()
    root = resolve(args.root)
    for name in args.experiments:
        config = experiment_config(name, root, args.steps)
        output = resolve(config["paths"]["output"])
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
        seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
        checkpoint = resolve(config["paths"]["save_checkpoint"])
        resume = checkpoint.is_file()
        artifact = load_checkpoint(checkpoint) if resume else None
        model, _, _ = build_model(
            config, config["device"], checkpoint_override=str(checkpoint) if resume else None
        )
        report = run_training(model, copy.deepcopy(config), args.steps, artifact, resume, status="trained")
        model.refresh_identity()
        write_json(output / "diagnostic_evaluation.json", diagnostic_reconstruction(model, config))
        print(json.dumps(dict(experiment=name, steps=report["completed_steps"], converged=False)), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
