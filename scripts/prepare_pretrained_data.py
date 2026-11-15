"""Make independent lossless RGB clips from the existing native 1080p YUV sources."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("train", "MCL-JCV/videoSRC01_1920x1080_30.yuv", "mcl_jcv01", 30, 64),
    ("validation", "UVG/Beauty_1920x1080_120fps_420_8bit_YUV.yuv", "uvg_beauty", 120, 16),
    ("test", "HEVC-B/BasketballDrive_1920x1080_50.yuv", "basketball", 50, 32),
)


def conversion_command(source, destination, fps, frames):
    return [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-video_size",
        "1920x1080",
        "-framerate",
        str(fps),
        "-i",
        str(source),
        "-vf",
        "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-preset",
        "fast",
        "-threads",
        "4",
        str(destination),
    ]


def inspect_clip(path, fps, frames):
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,pix_fmt,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(result)["streams"][0]
    expected = {
        "width": 1920,
        "height": 1080,
        "nb_frames": str(frames),
        "pix_fmt": "gbrp",
        "r_frame_rate": f"{fps}/1",
    }
    if info != expected:
        raise ValueError(f"unexpected prepared video metadata: {path}: {info}")
    return info


def prepare_source(source_root, output, spec):
    split, relative, name, fps, frames = spec
    source = source_root / relative
    size = source.stat().st_size
    if size % (1920 * 1080 * 3 // 2) or size < frames * 1920 * 1080 * 3 // 2:
        raise ValueError(f"invalid or insufficient raw YUV420 frames: {source}")
    destination = output / f"{name}.mp4"
    temporary = destination.with_suffix(".tmp.mp4")
    command = conversion_command(source, temporary, fps, frames)
    if not destination.exists():
        subprocess.run(command, check=True)
        inspect_clip(temporary, fps, frames)
        temporary.replace(destination)
    info = inspect_clip(destination, fps, frames)
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(
        split=split,
        source=str(source),
        source_bytes=size,
        source_frames=size // (1920 * 1080 * 3 // 2),
        path=str(destination),
        frames=frames,
        fps=fps,
        sha256=digest,
        metadata=info,
        command=command,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/whisperliang/video-tokenizer-benchmark/data/compression_sources"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/pretrained_native")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [prepare_source(args.source_root, args.output, spec) for spec in SOURCES]
    manifest = {
        "operation": "native pixels, no resize; lossless RGB storage after YUV420 conversion",
        "raw_color_assumption": "BT.709 limited-range YUV420p8; raw files carry no color metadata",
        "scope": "three independent clips for engineering training/validation/test; not benchmark-wide convergence",
        "records": records,
    }
    target = args.output / "manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
