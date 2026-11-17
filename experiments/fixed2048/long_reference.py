"""Post-hoc Validation rollout explanation; never trains or changes model selection."""

import argparse
import json
import time

import torch
from torch.nn import functional as F

from cp_adaptok.runtime import metadata, resolve, write_json
from experiments.fixed2048.backbone_control import continuous_intra_latent
from experiments.fixed2048.checkpoint import load_reconstructor, sha256
from experiments.fixed2048.cli import seeded_config
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import aggregate, distortion, grouped, psnr
from experiments.fixed2048.evaluate import FLAGS, advance_reference, force_intra


@torch.no_grad()
def rollout(model, clip, video, path, reference):
    state, rows = model.empty_state(), []
    for index, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode(rgb, state, index, force_intra=force_intra(reference, index))
        output = model.reconstruct(encoded, path)
        target = encoded.latent
        if encoded.context.frame_type == "I":
            target = continuous_intra_latent(model.intra, target, model.config.qp)
        direct, _ = model.decode_latent(target, encoded.context)
        mse = F.mse_loss(rgb, output.rgb).item()
        direct_mse = F.mse_loss(rgb, direct).item()
        memory = encoded.context.memory
        rows.append(
            {
                "video": video,
                "frame": index,
                "frame_type": encoded.context.frame_type,
                "path": path,
                "reference": reference,
                "group": f"{reference}/{path}",
                "mse": mse,
                "psnr": psnr(mse),
                "direct_mse": direct_mse,
                "direct_psnr": psnr(direct_mse),
                "feature_rms": output.feature.float().square().mean().sqrt().item(),
                "memory_rms": None if memory is None else memory.float().square().mean().sqrt().item(),
                "pixels": 1080 * 1920,
                "rgb_shape": list(output.rgb.shape),
                "posthoc": True,
                "used_for_selection": False,
                **distortion(target, output.latent),
                **FLAGS,
            }
        )
        state = advance_reference(model, encoded, {path: output}, [path], reference)
    return rows


def window_summary(rows, start, end):
    window = [r for r in rows if start <= r["frame"] < end and r["frame_type"] == "P"]
    if not window:
        raise ValueError("expected P window is missing from a temporal rollout")
    return {
        "psnr": psnr(sum(r["mse"] for r in window) / len(window)),
        "direct_psnr": psnr(sum(r["direct_mse"] for r in window) / len(window)),
        "feature_rms": sum(r["feature_rms"] for r in window) / len(window),
    }


def summarize(rows):
    result = {}
    for key, selected in grouped(rows, "group").items():
        summary = aggregate(selected)
        for label, start, end in [("early_P", 1, 8), ("late_P", 40, 48)]:
            summary[label] = None if summary["P"] is None else window_summary(selected, start, end)
        result[key] = summary
    return result


def diagnose_checkpoint(config, sources, hashes, name, checkpoint, directory, seal):
    model, identity = load_reconstructor(checkpoint, config["device"])
    model.requires_grad_(False)
    rows = []
    start = time.perf_counter()
    for source in sources:
        clip = read_clip(source, 48)
        for reference in ("i_only", "full2048", "matched128", "original128"):
            for path in ("R1", "R2"):
                rows.extend(rollout(model, clip, source["content_id"], path, reference))
    with (directory / f"{name}_frames.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(directory / f"{name}_summary.json", summarize(rows))
    write_json(
        directory / f"{name}_run.json",
        {
            "checkpoint": identity,
            "checkpoint_unchanged": sha256(checkpoint) == identity["sha256"],
            "manifest_hashes": hashes,
            "final_seal_sha256": sha256(seal),
            "posthoc_after_sealed_test": True,
            "used_for_selection": False,
            "frames_per_video": 48,
            "metric_names": ["mse", "psnr", "latent"],
            "direct_path": "same FrozenFrameContext, I retains official continuous scales; direct never commits",
            "completed_rows": len(rows),
            "wall_s": time.perf_counter() - start,
            **metadata(config, "complete"),
            **FLAGS,
        },
    )
    print("completed long48", name, len(rows), flush=True)


def require_posthoc_paths(directory):
    if directory.exists():
        raise FileExistsError("post-hoc diagnostic already exists")
    seal = resolve("results/fixed2048/final_test/seal.json")
    if not seal.exists():
        raise ValueError("this explicitly post-hoc diagnostic requires the existing final test seal")
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed2048.yaml")
    parser.add_argument("--output", default="results/fixed2048/reference_ablation/long48_posthoc")
    args = parser.parse_args()
    config = seeded_config(args.config)
    directory = resolve(args.output)
    seal = require_posthoc_paths(directory)
    sources, hashes = records(config, "validation", config["validation"]["videos"])
    directory.mkdir(parents=True)
    checkpoints = {
        "B1": "checkpoints/large_scale/formal_20261116/s4/B1/latest.pt",
        "R1": "checkpoints/fixed2048/continuous/main/best_validation.pt",
        "R2": "checkpoints/fixed2048/vq/main/best_validation.pt",
    }
    for name, checkpoint in checkpoints.items():
        diagnose_checkpoint(config, sources, hashes, name, checkpoint, directory, seal)


if __name__ == "__main__":
    main()
