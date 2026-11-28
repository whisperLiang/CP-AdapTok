"""Run an LD or HT-S clip through the official or progressive codec."""

import argparse
from functools import partial
from pathlib import Path

import cv2
import torch

from baselines.dcvc_uf.native import NativeNetworks
from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule
from common.artifacts import write_json
from common.data import require_unsealed_path
from common.performance import timed
from common.video import VideoSpec, read_video
from pht_tok.api import NATIVE_FRAMES, PHTTok, official_settings
from third_party.DCVC.src.utils.common import set_torch_env

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def input_clip(location: Path, count: int | None):
    path = require_unsealed_path(location)
    if path.is_dir():
        entries = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_TYPES)
        count = len(entries) if count is None else count
        for item in entries[:count]:
            require_unsealed_path(item)
        fps = 30.0
    else:
        reader = cv2.VideoCapture(str(path))
        try:
            if not reader.isOpened():
                raise ValueError(f"cannot open {path}")
            count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) if count is None else count
            fps = reader.get(cv2.CAP_PROP_FPS)
        finally:
            reader.release()
    if count < 1 or fps <= 0:
        raise ValueError("input needs a positive frame count and frame rate")
    return read_video(VideoSpec(str(path), count)), fps


def run_native(clip, units, checkpoint, skip, args):
    tx = NativeNetworks(checkpoint, args.backend, skip)
    rx = NativeNetworks(checkpoint, args.backend, skip)
    writer, reader = NativeNAL(), NativeNAL()
    for unit in units:
        def encode():
            return writer.encode(tx.encode(clip, unit, args.qp))

        payload, enc = timed(encode)
        _, dec = timed(lambda: rx.decode(reader.decode(payload, unit.start)))
        yield unit, payload, (len(payload),), enc, dec


def run_progressive(clip, units, checkpoint, args):
    tx = PHTTok(args.backend, checkpoint, args.device, layers=args.layers, reference_layers=args.reference_layers)
    for unit in units:
        encode = partial(tx.encode, qp=args.qp, intra=unit.intra, reset=unit.reset)
        packet, enc = timed(encode, clip[unit.start:unit.start + unit.valid])
        budget = len(packet.data) if args.budget_bytes is None else args.budget_bytes
        payload = packet.prefix(max_bytes=budget)
        required = 1 if unit.intra else tx.reference_layers
        if len(payload) < packet.prefix_boundaries[required - 1]:
            raise ValueError(f"frame {unit.start} requires {packet.prefix_boundaries[required - 1]} reference bytes")
        result, dec = timed(tx.decode, payload)
        if not result.reference_committed:
            raise RuntimeError("the required reference prefix was not committed")
        yield unit, payload, packet.prefix_boundaries, enc, dec


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=NATIVE_FRAMES, required=True)
    parser.add_argument("--mode", choices=("native", "progressive"), default="progressive")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qp", type=int, choices=range(64), metavar="0–63", default=32)
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True, help="directory for the physical access units")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--layers", type=int, choices=(4, 8))
    parser.add_argument("--reference-layers", type=int)
    args = parser.parse_args()
    if args.mode == "native" and any(value is not None for value in (args.budget_bytes, args.layers, args.reference_layers)):
        parser.error("budget and layer options apply to progressive mode")
    return args


@torch.inference_mode()
def main():
    args = arguments()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the official inference backends require CUDA")
    clip, fps = input_clip(args.input, args.frames)
    settings = official_settings()
    schedule = NativeSchedule(NATIVE_FRAMES[args.backend], reset_interval=settings["reset_interval"])
    units = list(schedule.units(len(clip)))
    checkpoint = args.checkpoint.parent if args.checkpoint.is_file() else args.checkpoint
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
        set_torch_env()
        clip = clip.to(device)
        measured = (run_native(clip, units, checkpoint, settings["skip_threshold"], args) if args.mode == "native"
                    else run_progressive(clip, units, checkpoint, args))
        outputs = list(measured)
    write_outputs(args, fps, outputs)


def write_outputs(args, fps, outputs):
    args.output.mkdir(parents=True, exist_ok=False)
    suffix = "bin" if args.mode == "native" else "pht"
    ledger = []
    for unit, payload, boundaries, enc, dec in outputs:
        name = f"{unit.start:08d}.{suffix}"
        (args.output / name).write_bytes(payload)
        row = {"file": name, "start": unit.start, "valid_frames": unit.valid, "bytes": len(payload),
               "complete_stream_boundaries": boundaries}
        if args.benchmark:
            row.update(encode_ms=enc, decode_ms=dec, capture_wait_ms=(unit.valid - 1) * 1000 / fps)
        ledger.append(row)
    write_json(args.output / "units.json", {"backend": args.backend, "mode": args.mode, "fps": fps, "units": ledger})
