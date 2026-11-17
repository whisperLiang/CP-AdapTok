"""Real-source engineering probes and independently measured single-clip learning."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptok_hr.codec.bitstream import parse, truncate_packet
from adaptok_hr.codec.codec import Codec
from adaptok_hr.config import ModelConfig
from adaptok_hr.data.dataset import ClipSampler, read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import save_image
from adaptok_hr.models.model import AdapTokHR
from adaptok_hr.runtime import environment, read_config, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model, save_training
from adaptok_hr.training.train import JointTrainer


def phase0(config, output):
    output = Path(output)
    if (output / "phase0.json").exists():
        raise ValueError("engineering probe is already complete; use another output to repeat")
    index = load_index(config["data"]["index"])
    source = next(r for r in index["splits"]["train"] if r["content_id"] == "DAVIS2017:bear")
    clip, metadata = read_clip(source, 0, 33)
    measured = {}
    for length in (1, 4, 16, 32):
        seed_everything(config["training"]["seed"])
        model = AdapTokHR(ModelConfig(**config["model"])).cuda()
        trainer = JointTrainer(model, config["training"])
        result = trainer.update(clip[:length], 1)
        if not all(v > 0 for v in result["module_update_norms"].values()):
            raise RuntimeError("a trainable model component did not update")
        sampler = ClipSampler([source], config["training"]["seed"] + 1)
        ckpt = save_training(f"checkpoints/adaptok_hr/phase0/length_{length}.pt", trainer, sampler,
                             config, index["identity"], 1, length)
        measured[str(length)] = {"update": result, "checkpoint": ckpt}
        del trainer, model
        torch.cuda.empty_cache()
        print(json.dumps({"length": length, "peak_gpu_mib": result["peak_gpu_mib"],
                          "update_seconds": result["update_seconds"]}), flush=True)
    model, _ = load_model("checkpoints/adaptok_hr/phase0/length_32.pt")
    codec = Codec(model.eval(), amp_bf16=True)
    sending, receiving, frames = None, None, []
    for frame_index, frame in enumerate(clip):
        frame = frame[None].cuda()
        packet, sending = codec.encode_frame(frame, sending, 2048, "phase0-bear")
        rgb, receiving = codec.decode_frame(packet, receiving)
        if not torch.equal(sending.rgb, rgb) or not torch.equal(sending.feature, receiving.feature):
            raise RuntimeError("causal receiver divergence")
        if frame_index in (0, 32):
            fresh, _ = codec.decode_frame(packet, None)
            torch.testing.assert_close(fresh, rgb, rtol=0, atol=0)
        if frame_index == 0:
            full = parse(packet).ids
            for budget in model.config.budgets:
                short = truncate_packet(packet, budget)
                assert parse(short).ids == full[:budget]
                codec.decode_frame(short, None)
            save_image(output / "phase0_original.png", frame)
            save_image(output / "phase0_reconstruction.png", rgb)
        frames.append({"index": frame_index, "frame_type": receiving.frame_type,
                       "bytes": len(packet), "state_chain": receiving.chain.hex()})
    write_json(output / "phase0.json", {"environment": environment(), "real_source": metadata,
                                        "training_probes": measured, "stream_frames": frames,
                                        "roundtrip_equal": True, "gop32_reset_equal": True,
                                        "training_probe_optimizer_steps": 4,
                                        "training_probe_frames": sum((1, 4, 16, 32))})


@torch.inference_mode()
def spatial_measure(model, source, output, amp_bf16=True):
    clip, _ = read_clip(source, 0, 4)
    codec = Codec(model.eval(), amp_bf16=amp_bf16)
    state, rows = None, []
    for index, frame in enumerate(clip):
        target = frame[None].cuda()
        _, state = codec.encode_frame(target, state, 2048, "fit-measurement")
        rgb = state.rgb.float()
        mse = F.mse_loss(rgb, target).item()
        dx = (rgb[..., 1:] - rgb[..., :-1]) - (target[..., 1:] - target[..., :-1])
        dy = (rgb[..., 1:, :] - rgb[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
        rows.append({"mse": mse, "edge_mse": .5 * (dx.square().mean() + dy.square().mean()).item(),
                     "output_spatial_std": rgb.std(dim=(-2, -1)).mean().item()})
        if index == 0:
            save_image(Path(output) / "original.png", target)
            save_image(Path(output) / "reconstruction.png", rgb)
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def fit_diagnostic(config, checkpoint, output):
    index = load_index(config["data"]["index"])
    source = next(r for r in index["splits"]["train"] if r["content_id"] == "DAVIS2017:bear")
    seed_everything(config["training"]["seed"])
    initial = AdapTokHR(ModelConfig(**config["model"])).cuda().eval()
    before = spatial_measure(initial, source, Path(output) / "initial")
    del initial
    trained, metadata = load_model(checkpoint)
    after = spatial_measure(trained, source, Path(output) / "trained")
    write_json(Path(output) / "fit_diagnostic.json", {"environment": environment(), "before": before, "after": after,
                "initialization": "reconstructed from recorded seed/config; not a historical pre-training measurement",
                "completed_steps": metadata["completed_steps"], "training_frames": metadata["training_frames"],
                "rgb_mse_improved": after["mse"] < before["mse"], "edge_mse_improved": after["edge_mse"] < before["edge_mse"],
                "scope": "four Train bear frames, real VQ2048 with reconstructed history; no generalization claim"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("phase0", "fit-diagnostic"))
    parser.add_argument("--config", default="configs/adaptok_hr/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/adaptok_hr/fit/step_000300.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = read_config(args.config)
    if args.mode == "phase0":
        phase0(config, args.output)
    else:
        fit_diagnostic(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
