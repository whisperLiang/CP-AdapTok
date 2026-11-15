"""Decode retained baseline packets with the updated implementation and strict historical references."""

import argparse
import struct
from pathlib import Path

import torch

from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.streaming.receiver import CodecDecoder


def wire_records(file):
    source, position, records = file.read_bytes(), 0, []
    while position < len(source):
        length = struct.unpack_from(">I", source, position)[0]
        position += 4
        records.append(source[position:position + length])
        position += length
    return records


def decode_records(receiver, records, all_prefixes):
    receiver.start_frame(records[0])
    if not all_prefixes:
        receiver.append_packets(records[1:])
        receiver.decode_prefix(receiver.model.config.n_max, use_cache=True)
        return 0.0
    maximum = 0.0
    for packet in records[1:]:
        k = receiver.append_packets([packet])
        if k < receiver.model.config.k_ref:
            continue
        dense = receiver.decode_prefix(k)
        cached = receiver.decode_prefix(k, use_cache=True)
        maximum = max(maximum, (dense - cached).abs().max().item())
        torch.testing.assert_close(cached, dense, atol=1e-5, rtol=1e-5)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results/optimization_20261115/historical_stream.json")
    parser.add_argument("--config", default="configs/pretrained_s1.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained_s1/step100.pt")
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], 0.8)
    model, _, _ = build_model(
        config, config["device"], checkpoint_override=args.checkpoint
    )
    model.eval()
    receiver = CodecDecoder(model)
    files = sorted(Path(args.source).glob("frame*.cpat"))
    if args.max_frames:
        files = files[:args.max_frames]
    if not files:
        raise ValueError("no historical packet files")
    with torch.no_grad():
        errors = [decode_records(receiver, wire_records(file), args.all_prefixes) for file in files]
    write_json(
        args.output,
        {
            "source": args.source,
            "frames": len(files),
            "strict_next_frame_reference_validation": True,
            "model_id": model.model_id,
            "format_changed": False,
            "decode_succeeded": True,
            "all_packet_prefixes_compared": args.all_prefixes,
            "cache_max_error": max(errors),
            "base_commits": receiver.base_commits,
        },
    )


if __name__ == "__main__":
    main()
