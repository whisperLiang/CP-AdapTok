"""Validation-only stopping decision and token-level diagnostics; no Test reads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import histogram_summary, psnr
from adaptok_hr.runtime import file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model


def layer_diagnostics(directory):
    histograms = [np.zeros(8192, dtype=np.int64) for _ in range(3)]
    boundaries = (0, 135, 645, 2048)
    frames = 0
    for path in sorted((directory / "k2048").rglob("*.bin")):
        packet = parse(path.read_bytes())
        for level, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            histograms[level] += np.bincount(packet.ids[start:stop], minlength=8192)
        frames += 1
    return {"frames": frames, "layers": {
        name: {**histogram_summary(hist), "most_frequent_fraction": float(hist.max() / hist.sum())}
        for name, hist in zip(("Base", "Medium", "Fine_transmitted_prefix"), histograms)}}


@torch.inference_mode()
def same_context_probe(model, index, evaluation):
    source = next(r for r in index["splits"]["validation"] if r["content_id"] == "DAVIS2017:car-roundabout")
    clip, _ = read_clip(source, 0, 64)
    codec = Codec(model.eval(), amp_bf16=True)
    rows, state = [], None
    for frame_index in range(64):
        packet = (evaluation / "k2048" / "DAVIS2017_car-roundabout" / f"frame_{frame_index:04d}.bin").read_bytes()
        previous = state
        full, state = codec.decode_frame(packet, state)
        if frame_index in (0, 1, 7, 31, 32, 63):
            target = clip[frame_index:frame_index + 1].cuda()
            ids = torch.tensor([parse(packet).ids], dtype=torch.long, device="cuda")
            measured = {}
            for budget in model.config.budgets:
                with codec.autocast():
                    rgb, _ = model.decode_ids(ids[:, :budget], previous)
                mse = (rgb.float() - target).square().mean().item()
                measured[str(budget)] = {"mse": mse, "psnr": psnr(mse),
                                         "rgb_difference_from_full_mse": (rgb.float() - full.float()).square().mean().item()}
            rows.append({"frame": frame_index, "frame_type": state.frame_type, "budgets": measured})
    return {"source": source["content_id"], "history": "actual K2048 binary stream; auxiliaries never advance state",
            "frames": rows, "scope": "same preceding state isolates current prefix effects, not independent-budget trajectories"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/adaptok_hr"))
    parser.add_argument("--checkpoint", default="checkpoints/adaptok_hr/joint/step_001000.pt")
    args = parser.parse_args()
    seed_everything(20261117)
    torch.set_num_threads(4)
    validation = json.loads((args.root / "validation_1000/summary.json").read_text())["budgets"]
    index = load_index("data/adaptok_hr/index.json")
    model, metadata = load_model(args.checkpoint)
    diagnostics = layer_diagnostics(args.root / "validation_1000")
    diagnostics["same_context"] = same_context_probe(model, index, args.root / "validation_1000")
    write_json(args.root / "token_diagnostics.json", diagnostics)
    last, medium = validation["2048"], validation["1024"]
    decision = {"status": "native_1080p_pilot_complete_targets_not_met", "completed_joint_steps": metadata["completed_steps"],
                "training_frames": metadata["training_frames"], "selected_checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": file_hash(args.checkpoint), "selected_before_test": True,
                "run_5000_updates": False, "scorer_ready": False, "convergence_verified": False,
                "validation_k256_to_k2048_gain_db": last["psnr_global"] - validation["256"]["psnr_global"],
                "validation_k1024_to_k2048_gain_db": last["psnr_global"] - medium["psnr_global"],
                "k2048_lpips": last["lpips"], "k2048_dists": last["dists"],
                "k2048_perplexity": last["perplexity"],
                "reason": "Stop after the bounded 1000-update pilot: perceptual quality remains poor, the final doubled prefix does not improve aggregate validation PSNR, and Fine-code usage is concentrated. Diagnose VQ/level contribution and decoder detail before committing a 5000-update or Scorer run.",
                "next_experiment": "controlled Train/Validation study of per-level codebook initialization/diversity and decoder spatial-detail capacity, with matching compute/data splits; larger licensed dataset still needed",
                "ablations_completed": False, "phase2_formal_training_completed": False}
    write_json(args.root / "decision.json", decision)
    print(json.dumps({k: v for k, v in decision.items() if k != "next_experiment"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
