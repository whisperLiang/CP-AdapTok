"""Bounded native training checks and a shared, untouched 1080p stream evaluation."""

import argparse
import copy
import json

import torch

from cp_adaptok.cli.eval import native_evaluation
from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.runtime import build_model, load_checkpoint, read_config, resolve, seed_everything, write_json
from cp_adaptok.training import run_training
from scripts.compare_prefix_training import calibrate_entropy
from scripts.evaluate_prefix_gain import summarize_reports

ROOT = "results/prefix_utilization"


def variant_sources():
    sources = [("original", "configs/prefix_diagnostics.yaml", "results/optimization_20261115/auxiliary/latest.pt")]
    for name in ("C0", "C1", "C2"):
        path = f"{ROOT}/decoder_ablations/{name}"
        sources.append((name, f"{path}/config.json", f"{path}/latest.pt"))
    sources.append(("fixed_spatial", "results/optimization_20261115/capacity/spatial_512/config.json",
                    "results/optimization_20261115/capacity/spatial_512/latest.pt"))
    return sources


def save_calibrated(model, config, source, output):
    calibration_config = copy.deepcopy(config)
    common = read_config(f"{ROOT}/full_1080p/evaluation_config.json")
    calibration_config["data"]["train"] = common["data"]["train"]
    calibration_config["training"].update(crop=512, sequence_length=4, fixed_sample_seed=19)
    calibration = calibrate_entropy(model, calibration_config, clips=len(common["data"]["train"]))
    calibration["training_specs"] = common["data"]["train"]
    artifact = load_checkpoint(source)
    artifact.update(model=model.state_dict(), entropy_calibration=calibration, inference_only=True)
    artifact.pop("optimizer", None)
    target = output / "calibrated.pt"
    torch.save(artifact, target)
    write_json(output / "entropy_calibration.json", calibration)
    model.artifact_provenance.update(entropy_calibrated_checkpoint=str(target),
                                     entropy_calibration=calibration)


def evaluate_variant(name, source_config, checkpoint, frames):
    common = read_config(f"{ROOT}/full_1080p/evaluation_config.json")
    config = read_config(source_config)
    config["data"]["test"] = [dict(spec, frames=frames) for spec in common["data"]["test"]]
    config["evaluation"].update(metrics=["psnr", "ms-ssim", "lpips", "dists"], save_frames=False)
    output = resolve(f"{ROOT}/full_1080p/{name}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=checkpoint)
    save_calibrated(model, config, checkpoint, output)
    evaluate_sequences(model.eval(), config, status="trained", output=output, split="test", compare_cache=True)
    del model
    torch.cuda.empty_cache()
    return str(output / "evaluation.json")


def native_controls(frames, qps):
    config = read_config(f"{ROOT}/full_1080p/evaluation_config.json")
    config["data"]["test"] = [dict(spec, frames=frames) for spec in config["data"]["test"]]
    config["evaluation"]["metrics"] = ["psnr", "ms-ssim", "lpips", "dists"]
    for qp in qps:
        config["model"]["qp"] = qp
        output = resolve(f"{ROOT}/full_1080p/native_qp{qp}")
        write_json(output / "config.json", config)
        native_evaluation(config, config["device"], output=output, split="test", max_frames=frames)


def engineering_curriculum(steps):
    """Only shape/state/training sanity; this never overrides the formal quality gate."""
    common = read_config(f"{ROOT}/full_1080p/evaluation_config.json")
    for name, source_config, source in variant_sources()[1:4]:
        for phase, crop, length in (("crop512", 512, 4), ("crop720", [720, 1280], 8),
                                    ("native1080", [1080, 1920], 16)):
            config = read_config(source_config)
            config["data"]["train"] = common["data"]["train"]
            config["stage"] = 2
            config["training"].update(crop=crop, sequence_length=length, tbptt=2,
                                         fixed_sample_seed=19, lr=0.00001,
                                         diagnostic_every=1000000, diagnostic_steps=[0], save_every=steps)
            output = resolve(f"{ROOT}/full_1080p/engineering_training/{name}/{phase}")
            target = output / "latest.pt"
            config["paths"].update(checkpoint=source, output=str(output), save_checkpoint=str(target))
            write_json(output / "config.json", config)
            seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
            resume = target.is_file()
            model, _, artifact = build_model(config, config["device"],
                                             checkpoint_override=str(target) if resume else source)
            run_training(model, config, steps, artifact, resume, status="trained")
            write_json(output / "scope.json", {"engineering_only": True, "formal_quality_gate_passed": False,
                       "purpose": "native pixel resolutions and 4/8/16-frame base-only TBPTT checks",
                       "updates": steps, "frames_per_update": length, "source_checkpoint": source})
            source = str(target)
            del model
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=["original", "C0", "C1", "C2", "fixed_spatial"])
    parser.add_argument("--native-qps", nargs="*", type=int, default=[16, 32, 48, 63])
    parser.add_argument("--engineering-steps", type=int, default=0)
    args = parser.parse_args()
    if args.engineering_steps:
        engineering_curriculum(args.engineering_steps)
    paths, labels = [], []
    for name, config, checkpoint in variant_sources():
        if name in args.variants:
            paths.append(evaluate_variant(name, config, checkpoint, args.frames))
            labels.append(name)
            print(json.dumps({"evaluated": name, "report": paths[-1]}), flush=True)
    if paths:
        summarize_reports(paths, labels, f"{ROOT}/full_1080p/comparison")
    native_controls(args.frames, args.native_qps)


if __name__ == "__main__":
    main()
