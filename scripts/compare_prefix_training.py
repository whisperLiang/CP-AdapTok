"""Matched finite-budget prefix ablations, with train-only entropy calibration."""

import argparse
import copy
import json
import time

import torch

from cp_adaptok.data import read_video, sample_training_spec
from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.runtime import build_model, load_checkpoint, read_config, resolve, seed_everything, write_json
from cp_adaptok.streaming.sender import ids_digest
from cp_adaptok.training import run_training
from scripts.run_optimization_experiments import diagnostic_reconstruction


def matched_config(name, root, steps, crop, checkpoint=None):
    config = read_config("configs/prefix_nested.yaml")
    output = resolve(root) / name
    config["paths"].update(output=str(output), save_checkpoint=str(output / "latest.pt"))
    if checkpoint:
        config["paths"]["checkpoint"] = checkpoint
    train = config["training"]
    train.update(crop=crop, sequence_length=4, tbptt=2, fixed_sample_seed=19, lr=0.0003,
                 save_every=max(1, steps // 2), validate_every=1000000,
                 diagnostic_every=max(1, steps // 2), diagnostic_steps=[0])
    config["data"]["train"] = [{"path": "data/pretrained_native/mcl_jcv01.mp4", "frames": 16,
                                "fps": 30, "purpose": "fixed-real-clip-prefix-pilot"}]
    config["data"]["validation"][0]["frames"] = 4
    settings = config["prefix_training"]
    settings["one_sided"] = name in ("B3", "scheduled", "C1", "C2")
    if name in ("B0", "C0"):
        config.pop("prefix_training")
    if name in ("B2", "B3", "C1", "C2"):
        # Inverse inclusion probability: all five budgets receive equal expected weight.
        settings["weights"] = {128: 0.6, 256: 1.8, 512: 1.8, 1024: 1.8, 2048: 0.6}
        settings["weighting"] = "balanced"
    if name == "scheduled":
        settings.update(weighting="scheduled", weight_schedule=[
            {"step": steps // 2, "weights": {128: 0.6, 256: 1.8, 512: 1.8, 1024: 1.8, 2048: 0.6}}])
    return config


@torch.no_grad()
def calibrate_entropy(model, config, clips=1):
    """Smoothed multinomial MLE on training tokens; no validation/test observations."""
    model.eval()
    model.refresh_identity()
    counts = torch.ones_like(model.entropy_model.logits, dtype=torch.float64)
    observations = 0
    for source in config["data"]["train"][:clips]:
        generator = torch.Generator().manual_seed(config["training"].get("fixed_sample_seed", 19))
        spec = sample_training_spec(source, config["training"]["sequence_length"], generator)
        clip = read_video(spec, config["training"]["crop"], generator)
        state = model.empty_state()
        for frame in clip:
            encoded = model.encode_full(frame[None].to(model.device), state)
            grouped = encoded.vq.ids.view(-1, model.config.packet_tokens)
            counts.scatter_add_(1, grouped, torch.ones_like(grouped, dtype=counts.dtype))
            _, feature = model.decode_ids(encoded.prefix(model.config.k_ref), encoded.context)
            state = encoded.context.commit(feature, ids_digest(encoded.prefix(model.config.k_ref))).detach()
            observations += grouped.numel()
    model.entropy_model.logits.copy_(counts.log().to(model.entropy_model.logits))
    model.refresh_identity()
    return {"method": "train-only multinomial MLE; additive-one smoothing", "observed_tokens": observations,
            "sequences": min(clips, len(config["data"]["train"])), "test_tokens_used": False}


def evaluate_variant(model, config, output):
    started = time.perf_counter()
    write_json(output / "training_clip_evaluation.json", diagnostic_reconstruction(model, config))
    calibration = calibrate_entropy(model, config)
    write_json(output / "entropy_calibration.json", calibration)
    # Retain a separate calibrated artifact; the representation optimizer is never resumed from it.
    artifact = load_checkpoint(config["paths"]["save_checkpoint"])
    artifact.update(model=model.state_dict(), entropy_calibration=calibration)
    artifact.pop("optimizer")
    torch.save(artifact, output / "calibrated.pt")
    report = evaluate_sequences(model.eval(), config, split="validation",
                                budgets=config["evaluation"]["budgets"],
                                metric_names=("psnr", "lpips", "dists"), status="trained",
                                output=output / "validation", compare_cache=True)
    write_json(output / "validation.json", report)
    return {"elapsed_s": time.perf_counter() - started, "calibration": calibration}


def run_variant(name, config, steps):
    output = resolve(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(config["paths"]["save_checkpoint"])
    previous_config = output / "config.json"
    if previous_config.exists() and read_config(previous_config) != config:
        raise ValueError(f"refusing to mix incompatible experiments: {output}")
    write_json(previous_config, config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    resume = checkpoint.is_file()
    model, _, artifact = build_model(config, config["device"],
                                     checkpoint_override=str(checkpoint) if resume else None)
    report = run_training(model, copy.deepcopy(config), steps, artifact, resume, status="trained")
    validation = evaluate_variant(model, config, output)
    summary = {"variant": name, "updates": report["completed_steps"], "training_elapsed_s": report["elapsed_s"],
               "evaluation": validation, "converged_claim": False}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=["B0", "B1", "B2", "B3", "scheduled"],
                        default=["B0", "B1", "B2", "B3", "scheduled"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--checkpoint")
    parser.add_argument("--root", default="results/prefix_utilization/nested_training")
    args = parser.parse_args()
    for name in args.variants:
        run_variant(name, matched_config(name, args.root, args.steps, args.crop, args.checkpoint), args.steps)


if __name__ == "__main__":
    main()
