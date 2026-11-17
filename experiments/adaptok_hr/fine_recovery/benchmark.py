"""Uninstrumented continuous native stream timings, including both decoders."""

import argparse
import json
from pathlib import Path

import torch

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import timed, verify_endpoints
from adaptok_hr.models.temporal_state import frame_context
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from cp_adaptok.metrics import summarize_latency

from .common import independent_codec


@torch.inference_mode()
def stream(model, clip, budget, warmup=5):
    sender = Codec(model.eval(), amp_bf16=True)
    receiver = independent_codec(model)
    for _ in range(warmup):
        sending, receiving = None, None
        for cpu in clip[:2]:
            packet, sending = sender.encode_frame(cpu[None].cuda(), sending, budget, "warmup")
            _, receiving = receiver.decode_frame(packet, receiving)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sending, receiving, rows = None, None, []
    for index, cpu in enumerate(clip):
        frame = cpu[None].cuda()

        def encode():
            with sender.autocast():
                return model.token_ids(frame, sending, budget)[0].cpu().tolist()

        ids, encode_ms = timed(encode, sender.device)
        next_index, frame_type, _ = frame_context(sending, model.config.gop)
        (packet, sending), local_ms = timed(lambda: sender.encode_ids(ids, sending, next_index, frame_type, "benchmark"), sender.device)
        (_, receiving), receiver_ms = timed(lambda: receiver.decode_frame(packet, receiving), sender.device)
        verify_endpoints(sending, receiving)
        if index in (0, 32):
            rgb, _ = receiver.decode_frame(packet, None)
            torch.testing.assert_close(rgb, receiving.rgb, atol=0, rtol=0)
        rows.append({"frame": index, "frame_type": frame_type, "encoder_ms": encode_ms,
                     "sender_ms": encode_ms + local_ms, "receiver_ms": receiver_ms,
                     "joint_ms": encode_ms + local_ms + receiver_ms, "bytes": len(packet),
                     "tokens": len(parse(packet).ids)})
    return {"frames": rows, "timing": {key: summarize_latency([r[key + "_ms"] for r in rows])
                                         for key in ("encoder", "sender", "receiver", "joint")},
            "bytes": sum(r["bytes"] for r in rows), "actual_bpp": sum(r["bytes"] for r in rows) * 8 / (len(rows) * 1080 * 1920),
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
            "sender_receiver_bitwise_equal": True, "gop33_reset_bitwise_equal": True,
            "independent_model_instances": True, "amp_bf16": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("benchmark output exists")
    torch.set_num_threads(4)
    seed_everything(20261118)
    model, checkpoint = load_model(args.checkpoint)
    index = load_index("data/adaptok_hr/index.json")
    source = next(r for r in index["splits"]["validation"] if r["content_id"] == "DAVIS2017:car-roundabout")
    clip, metadata = read_clip(source, 0, 64)
    budgets = {}
    for budget in args.budgets:
        budgets[str(budget)] = stream(model, clip, budget)
        print(json.dumps({"budget": budget, "joint": budgets[str(budget)]["timing"]["joint"]}), flush=True)
    write_json(output, {"environment": environment(), "checkpoint_sha256": file_hash(args.checkpoint),
                        "completed_steps": checkpoint["completed_steps"], "manifest_identity": index["identity"],
                        "input": metadata, "parameters": sum(p.numel() for p in model.parameters()), "budgets": budgets,
                        "scope": "single H100 sequential encoder+13bit packing+sender local decode+independent receiver unpack/decode; synchronized segment wall times; excludes read,H2D,metrics,disk,assertions; no overlap or profiling"})


if __name__ == "__main__":
    main()
