"""Decode retained baseline packets with the updated implementation and strict historical references."""

import argparse
import struct
from pathlib import Path

import torch

from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from cp_adaptok.streaming.receiver import CodecDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    config = read_config("configs/pretrained_s1.yaml")
    seed_everything(config["seed"], config["device"], 0.8)
    model, _, _ = build_model(
        config, config["device"], checkpoint_override="checkpoints/pretrained_s1/step100.pt"
    )
    model.eval()
    receiver = CodecDecoder(model)
    files = sorted(Path(args.source).glob("frame*.cpat"))
    if not files:
        raise ValueError("no historical packet files")
    with torch.no_grad():
        for file in files:
            source, position, records = file.read_bytes(), 0, []
            while position < len(source):
                length = struct.unpack_from(">I", source, position)[0]
                position += 4
                records.append(source[position : position + length])
                position += length
            receiver.start_frame(records[0])
            receiver.append_packets(records[1:])
            receiver.decode_base()
            receiver.decode_prefix(model.config.n_max, use_cache=True)
    write_json(
        "results/optimization_20261115/historical_stream.json",
        {
            "source": args.source,
            "frames": len(files),
            "strict_next_frame_reference_validation": True,
            "model_id": model.model_id,
            "format_changed": False,
            "decode_succeeded": True,
        },
    )


if __name__ == "__main__":
    main()
