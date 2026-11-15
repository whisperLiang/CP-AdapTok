"""Same-weights full causal-video FP32/FP64 comparison with strict per-peer fingerprints."""

import argparse
from dataclasses import replace

import torch

from cp_adaptok.data import read_video
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec
from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], 0.8)
    reference, _, _ = build_model(config, config["device"], checkpoint_override=args.checkpoint)
    reference.eval()
    candidate = CausalProgressiveCodec(replace(reference.config, attention_precision="float32"))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    candidate.to(reference.device).eval()
    models = [reference, candidate]
    senders, receivers = [CodecEncoder(m) for m in models], [CodecDecoder(m) for m in models]
    metrics = MetricSuite(reference.device, ["psnr", "lpips", "dists"])
    clip = read_video(dict(config["data"]["test"][0], frames=args.frames))
    records = []
    with torch.no_grad():
        for t, frame in enumerate(clip):
            rgb = frame[None].to(reference.device)
            encoded = [s.encode_frame(rgb, budgets=config["evaluation"]["budgets"]) for s in senders]
            row = {
                "frame": t,
                "frame_type": encoded[0].header.frame_type,
                "different_token_ids": sum(
                    a != b for a, b in zip(encoded[0].token_ids, encoded[1].token_ids)
                ),
                "prefixes": [],
            }
            for receiver, wire in zip(receivers, encoded):
                receiver.start_frame(wire.header_bytes)
                receiver.append_packets(wire.packets)
                receiver.decode_base()
            for k in config["evaluation"]["budgets"]:
                outputs = [r.decode_prefix(k, use_cache=False) for r in receivers]
                cache_errors = []
                for receiver, dense in zip(receivers, outputs):
                    cached = receiver.decode_prefix(k, use_cache=True)
                    cache_errors.append((cached - dense).abs().max().item())
                row["prefixes"].append(
                    {
                        "tokens": k,
                        "cross_precision_max_error": (outputs[0] - outputs[1]).abs().max().item(),
                        "cache_max_errors": cache_errors,
                        "fp64": metrics(rgb, outputs[0]),
                        "fp32": metrics(rgb, outputs[1]),
                    }
                )
            row["reference_feature_max_error"] = (
                (senders[0].state.feature - senders[1].state.feature).abs().max().item()
            )
            row["peer_fingerprints_equal"] = [
                s.state.fingerprint() == r.state.fingerprint() for s, r in zip(senders, receivers)
            ]
            if not all(row["peer_fingerprints_equal"]):
                raise RuntimeError("strict reference drift in precision comparison")
            records.append(row)
    write_json(
        args.output,
        {
            "checkpoint": args.checkpoint,
            "frames": len(records),
            "records": records,
            "cross_device_verified": False,
            "strict_validation_relaxed": False,
            "default_reference_precision": "float64",
            "metric_errors": metrics.errors,
        },
    )


if __name__ == "__main__":
    main()
