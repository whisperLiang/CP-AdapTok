"""Run Microsoft's entry point on the sealed-manifest Calibration selections."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from third_party.DCVC.src.utils.stream_helper import NalType, read_header, read_ip_remaining, read_sps_remaining

ROOT = Path(__file__).resolve().parents[2]
DCVC = ROOT / "third_party/DCVC"
QPS = (0, 16, 32, 48, 63)
DELAYS = {"ld": 1, "hts": 8, "htl": 8}


def account_stream(raw: bytes, frames: int, fps: float, delay: int) -> dict:
    if not raw or frames < 1 or delay < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("stream and source timing are required")
    data = BytesIO(raw)
    dimensions: dict[int, tuple[int, int]] = {}
    units = []
    valid = padding = i_bytes = p_bytes = sps_bytes = payload_bytes = 0
    height = width = 0
    while data.tell() < len(raw):
        offset = data.tell()
        header = read_header(data)
        kind, sps_id = header["nal_type"], header["sps_id"]
        if kind == NalType.NAL_SPS:
            sps = read_sps_remaining(data, sps_id)
            dimensions[sps_id] = sps["height"], sps["width"]
            sps_bytes += data.tell() - offset
            continue
        current = dimensions[sps_id]
        if valid and current != (height, width):
            raise ValueError("changing resolutions need a separate rate ledger")
        height, width = current
        qp, parallel, reset, payload = read_ip_remaining(data)
        intra = kind == NalType.NAL_I
        coded = 1 if intra else delay
        count = min(coded, frames - valid)
        if count <= 0 or not payload or (not valid and not intra):
            raise ValueError("unexpected I/P unit")
        size = data.tell() - offset
        i_bytes += size if intra else 0
        p_bytes += 0 if intra else size
        payload_bytes += len(payload)
        units.append(dict(start=valid, valid=count, coded=coded, intra=intra, qp=qp, reset=bool(reset),
                          offset=offset, payload_bytes=len(payload), bytes=size, parallel=parallel))
        valid += count
        padding += coded - count
    if valid != frames or i_bytes + p_bytes + sps_bytes != len(raw) or height * width <= 0:
        raise ValueError("stream disagrees with source frame or byte count")
    bits, duration = 8 * len(raw), frames / fps
    return dict(physical_bytes=len(raw), physical_bits=bits, payload_bytes=payload_bytes,
                header_bytes=len(raw) - payload_bytes, i_bytes=i_bytes, p_bytes=p_bytes, sps_bytes=sps_bytes,
                valid_frames=frames, tail_padding_frames=padding, resolution=[height, width],
                spatial_padding=[(-height) % 16, (-width) % 16], source_fps=fps, seconds=duration,
                bpp=bits / (height * width * frames), mbps=bits / duration / 1e6, units=units,
                sha256=hashlib.sha256(raw).hexdigest())


def prepare(work: Path) -> dict:
    from scripts.hla_tok.uf.inputs import inputs

    reader, plans = inputs()
    sources = {}
    for clip in plans["calibration"]:
        record = reader.catalog("train")[clip.video]
        name = clip.video.split(":")[-1]
        frames = work / "rgb" / name
        frames.mkdir(parents=True, exist_ok=True)
        for i in range(clip.length):
            item = record["frame_hashes"][clip.start + i]
            raw = reader._open_allowed(record, item)
            (frames / f"im{i + 1:05}.png").write_bytes(raw)
        sources[name] = clip.manifest(reader) | {
            "width": 1920, "height": 1080, "frames": clip.length, "intra_period": -1,
            "original_frames": record["original_frames"], "fps_nominal": record["fps_nominal"],
        }
    return sources


def run(work: Path, weights: Path, sources: dict, models: list[str], repeat: bool,
        dataset: str = "Calibration") -> dict:
    src_type, base_path = ("png", "rgb") if dataset == "Calibration" else ("yuv420", "media")
    config = {"root_path": str(work), "test_classes": {
        dataset: {"test": 1, "base_path": base_path, "src_type": src_type, "sequences": sources}}}
    config_path = work / f"{dataset}.json"
    config_path.write_text(json.dumps(config))
    runs: dict[str, dict] = {}
    for model in models:
        key = f"{dataset}-{model}"
        directory = work / key
        output = directory / "official.json"
        qps = [str(q) for q in QPS]
        command = [sys.executable, str(DCVC / "test_video.py"),
                   "--model_path_i", str(weights / "cvpr2026_image.pth.tar"),
                   "--model_path_p", str(weights / f"cvpr2026_video_{model}.pth.tar"),
                   "--model_structure", model, "--rate_num", str(len(qps)), "--qp_i", *qps, "--qp_p", *qps,
                   "--test_config", str(config_path), "--output_path", str(output),
                   "--cuda_idx", "0", "-w", "1", "--skip_thres", "0.15", "--force_intra", "0",
                   "--reset_interval", "128", "--verbose", "2", "--verbose_json", "1", "--calc_ssim", "0",
                   "--stream_path", str(directory / "streams")]
        if repeat or not output.exists():
            subprocess.run(command, cwd=DCVC, check=True, env=os.environ.copy())
        runs[model] = summarize_run(output, command, sources, model, dataset)
    return {"sources": sources, "models": runs, "test_pixel_reads": 0, "validation_pixel_reads": 0}


def summarize_run(output: Path, command: list[str], sources: dict, model: str, dataset: str) -> dict:
    metrics = json.loads(output.read_bytes())[dataset]
    results = {}
    for qp_index, qp in enumerate(QPS):
        points = []
        for name, source in sources.items():
            raw = (output.parent / "streams" / dataset / f"{name}_q{qp}.bin").read_bytes()
            point = account_stream(raw, source["frames"], source["fps"], DELAYS[model])
            official = metrics[name][f"{qp_index:03}"]
            if abs(point["bpp"] - official["ave_all_frame_bpp"]) > 1e-6:
                raise ValueError("physical stream and official bpp differ")
            points.append({"source": name, **point, "official": official})
        results[str(qp)] = points
    return {"command": command, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", choices=DELAYS, nargs="+", default=list(DELAYS))
    parser.add_argument("--uvg", type=Path)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    if args.uvg:
        src = args.uvg.resolve()
        if src.parent != work / "media" or src.stat().st_size < 129 * 3110400:
            raise ValueError("the locked UVG sample must contain at least 129 complete frames")
        sources = {src.name: {"width": 1920, "height": 1080, "frames": 129, "fps": 120,
                              "intra_period": -1, "source": "UVG ShakeNDry 1080p 8-bit YUV420"}}
        report = run(work, args.weights.resolve(), sources, args.models, args.repeat, "UVG")
    else:
        report = run(work, args.weights.resolve(), prepare(work), args.models, args.repeat)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
