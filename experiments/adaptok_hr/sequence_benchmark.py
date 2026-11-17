"""Measure a continuous native stream without loading perceptual metric networks."""

import argparse
import json

import torch

from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import evaluate_source
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model


@torch.inference_mode()
def warmup(model, source, budget, repeats):
    clip, _ = read_clip(source, 0, 2)
    codec = Codec(model.eval(), amp_bf16=True)
    for _ in range(repeats):
        sender, receiver = None, None
        for frame in clip:
            packet, sender = codec.encode_frame(frame[None].cuda(), sender, budget, source["content_id"])
            _, receiver = codec.decode_frame(packet, receiver)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--content-id", default="DAVIS2017:car-roundabout")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 64 or args.warmup < 1:
        raise ValueError("sequence throughput requires at least 64 frames and a positive warmup")
    torch.set_num_threads(4)
    seed_everything(20261117)
    model, _ = load_model(args.checkpoint)
    index = load_index(args.index)
    source = next(r for r in index["splits"]["validation"] if r["content_id"] == args.content_id)
    if source["original_frames"] < args.frames:
        raise ValueError("source is shorter than requested continuous sequence; no frame padding")
    budgets = {}
    for budget in model.config.budgets:
        warmup(model, source, budget, args.warmup)
        summary, rows = evaluate_source(model, source, budget, args.frames, metrics=None)
        budgets[str(budget)] = {"summary": summary, "frames": rows, "sender_receiver_equal": True}
        print(json.dumps({"budget": budget, "frames": len(rows),
                          "joint": summary["sender_plus_receiver_latency"]}), flush=True)
    write_json(args.output, {"environment": environment(), "checkpoint_sha256": file_hash(args.checkpoint),
                            "manifest_identity": index["identity"], "warmup_I_P_pairs": args.warmup,
                            "batch_size": 1, "budgets": budgets,
                            "scope": "continuous real 1080p validation frames, synchronized per-frame wall; sender includes local decode and packing, receiver includes unpacking; excludes load,H2D,disk,metrics and endpoint equality checks; no perceptual network resident; no pipeline overlap"})


if __name__ == "__main__":
    main()
