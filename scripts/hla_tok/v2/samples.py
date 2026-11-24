"""Matched native PNG, crop and lossless video evidence for measured arms."""

# ruff: noqa: E402
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

VIDEO = "DAVIS2017:car-roundabout"
START = 32
FRAMES = 16


def write_clip(directory, label, images):
    from hla_tok.evaluation.metrics import sha256_file
    from hla_tok.evaluation.reconstruction import save_native
    from scripts.hla_tok.export_validation_video import video_writer

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.mkv"
    count = 0
    with video_writer(path, 30) as stream:
        for index, image in enumerate(images):
            if tuple(image.shape) != (1, 3, 1080, 1920):
                raise ValueError("sample video must contain native RGB")
            if index == 0:
                save_native(directory, label, image)
            pixels = image[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu()
            stream.write(pixels.numpy().tobytes())
            count += 1
    if count != FRAMES:
        raise ValueError("incomplete matched clip")
    return {"video": VIDEO, "start": START, "frames": count, "fps": 30, "resolution": [1080, 1920],
            "mkv_sha256": sha256_file(path), "png_sha256": sha256_file(directory / f"{label}.png")}


def normalized(reader):
    from hla_tok.evaluation.benchmark import transfer

    for frame in range(START, START + FRAMES):
        yield transfer(reader.read("validation", VIDEO, frame)[None])


def pilot_clip(reader, arm, path):
    import torch

    from hla_tok import HLATok
    from hla_tok.evaluation.metrics import sha256_file
    from hla_tok.training.v2 import load_experimental_checkpoint
    from scripts.hla_tok.v2.protocol import COHORT, FIXED_BUDGET

    model = HLATok(arm).cuda().eval().to(memory_format=torch.channels_last)
    load_experimental_checkpoint(path, model, sha256_file(COHORT))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for source in normalized(reader):
            encoded = model.encode_training(source, FIXED_BUDGET, quantize=False)
            yield model.decode_training(encoded, FIXED_BUDGET)


def pilot_samples(reader, directory, checkpoints, arms, step):
    from hla_tok.evaluation.metrics import sha256_file

    manifest = {}
    for arm in arms:
        path = checkpoints / f"continuous_pilot/seed1/{arm}/pilot/step_{step:05d}.pt"
        manifest[arm] = write_clip(directory, arm, pilot_clip(reader, arm, path)) | {
            "representation": "continuous K16320 diagnostic", "step": step, "checkpoint_sha256": sha256_file(path)
        }
    return manifest


def legacy_sample(directory, legacy_root):
    from hla_tok.evaluation.metrics import sha256_file
    from scripts.hla_tok.v2.protocol import OUTPUT, json_at, verified_checkpoints

    archive = legacy_root / "results/hla_tok/phase1/video"
    record = json_at(archive / "video_manifest.json")
    movie = archive / "K16320.mkv"
    checkpoint = verified_checkpoints(legacy_root)["fixed"]
    if (record["video"], record["start_frame"], record["frames"], record["fps"]) != (VIDEO, START, FRAMES, 30):
        raise ValueError("the archived R0 clip covers different frames")
    if sha256_file(checkpoint) != record["checkpoint_sha256"] or sha256_file(movie) != record["video_sha256"][movie.name]:
        raise ValueError("the archived R0 clip or checkpoint differs from its manifest")
    first_frame = OUTPUT / "diag0/replay_vq_post_refactor/samples"
    for source, target in ((movie, "R0_fixed.mkv"), (first_frame / "K16320.png", "R0_fixed.png"),
                           (first_frame / "K16320_crop.png", "R0_fixed_crop.png")):
        shutil.copyfile(source, directory / target)
    return {"video": VIDEO, "start": START, "frames": FRAMES, "fps": 30, "resolution": [1080, 1920],
            "representation": "actual Fixed13 payload", "checkpoint_sha256": record["checkpoint_sha256"],
            "actual_bpp": record["codec_rate"]["actual_bpp"], "mkv_sha256": record["video_sha256"][movie.name],
            "png_sha256": sha256_file(directory / "R0_fixed.png")}


def official_clip(reader, legacy_root, official_checkpoint):
    import torch

    from experiments.original_adaptok_1080p.tiling import TileLayout
    from hla_tok.evaluation.official import OfficialBaseline, reconstruct
    from scripts.hla_tok.v2.protocol import json_at

    pinned = json_at(ROOT / "results/original_adaptok_1080p/checkpoint_manifest.json")
    baseline = OfficialBaseline(legacy_root / "third_party/AdapTok", official_checkpoint, pinned)
    raw = torch.stack([reader.read("validation", VIDEO, step) for step in range(START, START + FRAMES)])
    with torch.inference_mode():
        _, outputs, _ = reconstruct(raw, baseline, TileLayout(1080, 1920, stride=128), (480,), 4)
    for output in outputs[480]:
        yield output[None]


def joint_clip(reader, directory, joint):
    import torch
    from aip_vtok.decoder.causal_state import ReferenceState
    from aip_vtok.joint import protocol
    from aip_vtok.joint.bitstream import FRAME_HEADER
    from aip_vtok.joint.evaluation import load_selection

    from scripts.hla_tok.v2.protocol import csv_at

    directory.mkdir(parents=True, exist_ok=True)
    protocol.OUTPUT = directory
    frozen = joint / "results/aip_vtok/joint_v1/seed1/validation"
    archived = {int(r["frame"]): r for r in csv_at(frozen / "validation_per_frame.csv") if
                r["video"] == VIDEO and r["method"] == "Joint V1" and r["history"] == "CLOSED" and r["budget"] == "16320"}
    with torch.no_grad():
        run = protocol.open_experiment(joint / protocol.CONFIG, seed=1)
        load_selection(run, 1)
        codec = run.codec
        reference, profile = ReferenceState(frame_id=START - 1), codec.profile()
        for number, source in zip(range(START, START + FRAMES), normalized(reader)):
            stream = frozen / "streams" / VIDEO.replace(":", "_") / "CLOSED" / f"{number:04d}.aipj"
            wire = stream.read_bytes()
            session = codec.begin_frame(reference, wire[:FRAME_HEADER.size], profile)
            used = FRAME_HEADER.size
            for packet in profile.wire.packets(wire):
                session.append_packet(packet)
                used += len(packet)
                if session.received == session.reference_budget:
                    reference = session.commit_reference()
                if session.received == 16320:
                    break
            item = archived[number]
            if used != int(item["total_bytes"]):
                raise ValueError("archived Joint V1 sample bytes differ")
            expected = item["rx_reference_sha256"]
            if expected and expected != reference.fingerprint():
                raise ValueError("archived Joint V1 reference differs")
            image = session.render(16320)
            if abs((image.float() - source).square().mean().item() - float(item["mse"])) > 1e-8:
                raise ValueError("archived Joint V1 sample reconstruction differs")
            yield image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilots", "extension", "legacy", "official", "joint"))
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--official-checkpoint", type=Path)
    parser.add_argument("--joint-root", type=Path)
    args = parser.parse_args()
    if args.mode == "joint":
        sys.path.insert(0, str(args.joint_root))

    from hla_tok.evaluation.io import write_json
    from scripts.hla_tok.v2.protocol import CHECKPOINTS, OUTPUT, audit_read_counts, frozen_data, setup

    setup()
    reader, _, _ = frozen_data()
    directory = OUTPUT / "samples" / args.mode
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    try:
        if args.mode == "pilots":
            manifest["source"] = write_clip(directory, "source", normalized(reader))
            manifest.update(pilot_samples(reader, directory, CHECKPOINTS, ("R0", "R1", "R2", "R3"), 1000))
        elif args.mode == "extension":
            manifest.update(pilot_samples(reader, directory, CHECKPOINTS, ("R0", "R2"), 2000))
        elif args.mode == "legacy":
            manifest["R0_fixed"] = legacy_sample(directory, args.legacy_root)
        elif args.mode == "official":
            manifest["original_adaptok"] = write_clip(directory, "original_adaptok", official_clip(
                reader, args.legacy_root, args.official_checkpoint)) | {"stride": 128, "tokens_per_tile_block": 480}
        else:
            manifest["joint_v1"] = write_clip(directory, "joint_v1", joint_clip(reader, directory, args.joint_root))
        write_json(directory / "manifest.json", manifest)
    finally:
        audit_read_counts(directory, reader)


if __name__ == "__main__":
    main()
