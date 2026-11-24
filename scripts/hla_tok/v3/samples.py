"""Matched native PNG/crop and FFV1 samples from actual checkpoints."""

import subprocess
from contextlib import ExitStack

import torch
from PIL import Image

from hla_tok import HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from scripts.hla_tok.export_validation_video import video_writer
from scripts.hla_tok.v2.samples import FRAMES, START, VIDEO
from scripts.hla_tok.v3.quality import _native_reconstructions


def _rgb24(image):
    return image[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes()


def _verify_first_frame(output, label):
    image = output / f"{label}.png"
    clip = output / f"{label}.mkv"
    with Image.open(image) as decoded:
        rgb = decoded.convert("RGB").tobytes()
    replay = subprocess.check_output(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip),
                                      "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    if rgb != replay:
        raise RuntimeError("FFV1 first frame differs from the original native PNG")
    return {"native_png_sha256": sha256_file(image), "ffv1_sha256": sha256_file(clip)}


@torch.inference_mode()
def export_samples(model, reader, cohort, checkpoint, mode, output, *, budget):
    if cohort[VIDEO] < START + FRAMES:
        raise RuntimeError("frozen native Validation does not contain the registered sample clip")
    output.mkdir(parents=True, exist_ok=False)
    before = dict(reader.counts)
    budgets = (budget,) if isinstance(model, HLATok) else ()
    try:
        with ExitStack() as stack, torch.autocast("cuda", dtype=torch.bfloat16):
            streams = {tag: stack.enter_context(video_writer(output / f"{tag}.mkv", 30))
                       for tag in ("source", "reconstruction")}
            for index in range(START, START + FRAMES):
                source = transfer(reader.read("validation", VIDEO, index)[None])
                _, decoded, _, _, _ = next(_native_reconstructions(model, source, budgets, mode == "fixed13", False))
                for tag, image in (("source", source), ("reconstruction", decoded)):
                    if index == START:
                        save_native(output, tag, image)
                    streams[tag].write(_rgb24(image))
        record = {"mode": mode, "checkpoint_sha256": sha256_file(checkpoint), "budget": budget if budgets else None,
                  "video": VIDEO, "start": START, "frames": FRAMES, "fps": 30,
                  "crop_xywh": [768, 432, 384, 216],
                  "samples": {tag: _verify_first_frame(output, tag) for tag in ("source", "reconstruction")}}
        write_json(output / "manifest.json", record)
        return record
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
