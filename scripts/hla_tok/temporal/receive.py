"""Render HLT prefix files with only the static assets and the frozen synthesis weights."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.streaming.temporal import TemporalReceiver
from hla_tok.temporal.native_wire import NativeWire
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.temporal.inputs import Assets, Setting
from scripts.hla_tok.v2.protocol import json_at, setup


def rgb_digest(pixels: torch.Tensor) -> str:
    return sha256(pixels.float().cpu().contiguous().numpy().astype("<f4", copy=False)).hexdigest()


@torch.inference_mode()
def receive(assets_path: Path, checkpoint: Path, setting_path: Path, packet_paths: list[Path],
            output: Path, precision: str, wire_library: Path | None = None) -> None:
    setup()
    assets = Assets(assets_path)
    profile = assets.profile(Setting(**json_at(setting_path)))
    synth = assets.receive_only(checkpoint)
    wire = NativeWire(wire_library, profile) if wire_library is not None else None
    receiver = TemporalReceiver(profile, wire=wire)
    rows = []
    for path in packet_paths:
        raw = path.read_bytes()
        frame = receiver.receive_stream(raw)
        state = receiver.commit_base(frame)
        display = receiver.render_prefix(frame)
        with compute(precision):
            pixels = synth.render(display)
        rows.append({"frame": frame, "bytes": len(raw), "wire_sha256": sha256(raw).hexdigest(),
                     "base_sha256": state.digest.hex(), "rgb_float32_sha256": rgb_digest(pixels),
                     "shape_nchw": list(pixels.shape)})
        receiver.finalize_frame(frame)
    write_json(output, {"status": "PASS", "precision": precision, "frames": rows,
                        "has_analysis_modules": hasattr(synth, "encoder") or hasattr(synth, "query_encoder")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--setting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--wire-library", type=Path)
    parser.add_argument("packets", nargs="+", type=Path)
    args = parser.parse_args()
    receive(args.assets, args.checkpoint, args.setting, args.packets, args.output, args.precision, args.wire_library)
