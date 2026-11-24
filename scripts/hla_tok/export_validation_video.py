"""Export a native lossless Validation clip reconstructed from actual Fixed13 payloads."""

# ruff: noqa: E402
import argparse
import json
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from hla_tok import Geometry, HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.fixed13 import pack_tokens_13bit, payload_ledger, unpack_tokens_13bit
from hla_tok.training.checkpoint import read_checkpoint


@contextmanager
def video_writer(path, fps):
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "1920x1080", "-framerate", str(fps), "-i", "pipe:0", "-threads", "4",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            yield encoder.stdin
        finally:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError(f"ffmpeg failed to write {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", default="DAVIS2017:car-roundabout")
    parser.add_argument("--start", type=int, default=32)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--budget", type=int, default=16320)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    cohort_file = ROOT / "configs/hla_tok/validation_cohort.json"
    identity = json.loads(cohort_file.read_bytes())
    reader = AuditedFrames(ROOT / "data/manifests", identity["manifest_hashes"])
    cohort = load_cohort(cohort_file, reader)
    if args.start < 0 or args.frames <= 0 or args.start + args.frames > cohort[args.video]:
        raise ValueError("clip must lie within the matched Validation cohort")
    checkpoint = read_checkpoint(args.checkpoint).state
    if checkpoint["identity"]["stage"] != "fixed" or checkpoint["identity"]["dataset_sha256"] != sha256_file(cohort_file):
        raise ValueError("video export requires a fixed-budget checkpoint on the matched dataset")
    sender, receiver = [HLATok().cuda().to(memory_format=torch.channels_last).eval() for _ in range(2)]
    for model in (sender, receiver):
        model.load_state_dict(checkpoint["model"])
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = args.output / "source.mkv"
    hla_path = args.output / f"K{args.budget}.mkv"
    meta = reader.catalog("validation")[args.video]
    budget = args.budget
    total_bytes = 0
    try:
        with ExitStack() as stack, torch.autocast("cuda", dtype=torch.bfloat16):
            src = stack.enter_context(video_writer(source_path, meta["fps"]))
            rec = stack.enter_context(video_writer(hla_path, meta["fps"]))
            for index in range(args.start, args.start + args.frames):
                rgb = reader.read("validation", args.video, index)
                src.write(rgb.permute(1, 2, 0).contiguous().numpy().tobytes())
                ids = sender.encode_frame(transfer(rgb[None]), budget)
                wire = pack_tokens_13bit(ids[0].tolist())
                payload = torch.tensor([unpack_tokens_13bit(wire, budget)], device="cuda", dtype=torch.long)
                reconstruction = receiver.decode_prefix(payload, Geometry(1080, 1920))
                pixels = reconstruction[0].mul(255).round().byte().permute(1, 2, 0).contiguous().cpu()
                rec.write(pixels.numpy().tobytes())
                total_bytes += len(wire)
    finally:
        write_json(args.output / "pixel_reads.json", {f"{k}_pixel_reads": v for k, v in reader.counts.items()})
    write_json(args.output / "video_manifest.json", {
        "video": args.video, "start_frame": args.start, "frames": args.frames,
        "width": 1920, "height": 1080, "fps": meta["fps"], "ffv1_lossless_rgb": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "video_sha256": {path.name: sha256_file(path) for path in (source_path, hla_path)},
        "codec_rate": payload_ledger(total_bytes, args.frames * budget, args.frames, 1080, 1920),
    })


if __name__ == "__main__":
    main()
