"""Matched native PNG/crop and FFV1 samples from actual checkpoints."""

import subprocess
from contextlib import ExitStack

import torch
from PIL import Image, ImageDraw

from hla_tok import BUDGETS, HLATok
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
    return {"native_png_sha256": sha256_file(image), "ffv1_sha256": sha256_file(clip),
            "crop_sha256": sha256_file(output / f"{label}_crop.png")}


def _progressive_sheet(output, labels):
    width, height = 384, 216
    row_height = height + 24
    sheet = Image.new("RGB", (3 * width, 3 * row_height), "white")
    text = ImageDraw.Draw(sheet)
    for index, label in enumerate(labels):
        left, top = index % 3 * width, index // 3 * row_height
        with Image.open(output / f"{label}_crop.png") as image:
            if image.size != (width, height):
                raise ValueError("progressive comparison requires unscaled native crops")
            sheet.paste(image, (left, top + 24))
        text.text((left + 8, top + 5), label, fill="black")
    destination = output / "progressive_native_crops.png"
    sheet.save(destination)
    return destination


def _sample_images(model, source, budgets, mode, progressive):
    yield "source", source
    for (_, budget), image, _, _, _ in _native_reconstructions(model, source, budgets, mode == "fixed13", False):
        yield (f"K{budget}" if progressive else "reconstruction"), image


def _write_sample_frames(model, reader, output, mode, budgets, labels, progressive):
    with ExitStack() as stack, torch.autocast("cuda", dtype=torch.bfloat16):
        streams = {tag: stack.enter_context(video_writer(output / f"{tag}.mkv", 30)) for tag in labels}
        for index in range(START, START + FRAMES):
            source = transfer(reader.read("validation", VIDEO, index)[None])
            for tag, image in _sample_images(model, source, budgets, mode, progressive):
                if index == START:
                    save_native(output, tag, image)
                streams[tag].write(_rgb24(image))


@torch.inference_mode()
def export_samples(model, reader, cohort, checkpoint, mode, output, *, budget, progressive=False):
    if cohort[VIDEO] < START + FRAMES:
        raise RuntimeError("frozen native Validation does not contain the registered sample clip")
    if progressive and mode != "fixed13":
        raise ValueError("progressive samples require true Fixed13 reconstruction")
    output.mkdir(parents=True, exist_ok=False)
    before = dict(reader.counts)
    budgets = BUDGETS if progressive else ((budget,) if isinstance(model, HLATok) else ())
    labels = ["source", *(f"K{value}" for value in budgets)] if progressive else ["source", "reconstruction"]
    try:
        _write_sample_frames(model, reader, output, mode, budgets, labels, progressive)
        record = {"mode": mode, "checkpoint_sha256": sha256_file(checkpoint), "budget": budget if budgets else None,
                  "video": VIDEO, "start": START, "frames": FRAMES, "fps": 30,
                  "crop_xywh": [768, 432, 384, 216],
                  "samples": {tag: _verify_first_frame(output, tag) for tag in labels}}
        if progressive:
            sheet = _progressive_sheet(output, labels)
            record |= {"budgets": list(budgets), "native_crop_comparison": sheet.name,
                       "native_crop_comparison_sha256": sha256_file(sheet)}
        write_json(output / "manifest.json", record)
        return record
    finally:
        write_json(output / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
