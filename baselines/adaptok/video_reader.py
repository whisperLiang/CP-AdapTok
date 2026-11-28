"""CP-AdapTok's OpenCV BGR->RGB rule; no resize, explicit raw-YUV conversion."""

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from common.data import dataset_path, require_unsealed_path
from common.integrity import sha256_file as file_sha256
from common.video import VideoSpec, read_video


def source_metadata(path, fps=30):
    path = require_unsealed_path(Path(path))
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"empty image sequence: {path}")
        hashes = [{"file": p.name, "sha256": file_sha256(p)} for p in images]
        digest = hashlib.sha256(b"".join(bytes.fromhex(row["sha256"]) for row in hashes)).hexdigest()
        image = cv2.imread(str(images[0]))
        h, w = image.shape[:2]
        return {"path": str(path), "sha256": digest, "frame_hashes": hashes, "width": w, "height": h,
                "fps": fps, "fps_nominal": True, "source_frames": len(images)}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    metadata = {"path": str(path), "sha256": file_sha256(path), "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": capture.get(cv2.CAP_PROP_FPS),
                "source_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), "fps_nominal": False}
    capture.release()
    return metadata


def read_yuv(path, frames, start, width, height, fps, pixel_format, color_range, color_matrix):
    if min(width, height, fps) <= 0:
        raise ValueError("raw YUV needs explicit dimensions and fps")
    # Explicitly declare limited/full range and BT.601/709; FFmpeg never rescales the image.
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", pixel_format,
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", str(path),
               "-vf", f"trim=start_frame={start}:end_frame={start+frames},scale=in_range={color_range}:out_range=full:in_color_matrix={color_matrix}",
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    data = subprocess.run(command, check=True, capture_output=True).stdout
    if len(data) != frames * height * width * 3:
        raise ValueError("raw YUV ended before the requested real frames")
    array = np.frombuffer(data, dtype=np.uint8).copy().reshape(frames, height, width, 3)
    return torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255


def read_input(path, frames=16, start=0, fps=30, raw_yuv=None, metadata=None):
    begin = time.perf_counter()
    path = require_unsealed_path(Path(path))
    if path.suffix.lower() == ".yuv":
        if raw_yuv is None:
            raise ValueError("raw YUV requires width, height, pixel_format, color_range and color_matrix")
        clip = read_yuv(path, frames, start, fps=fps, **raw_yuv)
        metadata = {"path": str(path), "sha256": file_sha256(path), "fps": fps, **raw_yuv}
    else:
        metadata = dict(metadata or source_metadata(path, fps))
        clip = read_video(VideoSpec(str(path), frames=frames, start=start, fps=metadata["fps"]))
    metadata = {**metadata, "start": start, "frames": frames, "read_ms": (time.perf_counter() - begin) * 1000,
                "shape": list(clip.shape), "rgb_sha256": hashlib.sha256(clip.numpy().tobytes()).hexdigest(),
                "rgb_conversion": "ffmpeg explicit range/matrix" if raw_yuv else "common.video.read_video; OpenCV BGR to RGB; float32 /255",
                "resize": False}
    if tuple(clip.shape) != (frames, 3, 1080, 1920) or not bool(torch.isfinite(clip).all()):
        raise ValueError(f"expected real native T,3,1080,1920 RGB frames, got {tuple(clip.shape)}")
    return clip, metadata


def manifest_sources(path, content_ids):
    manifest = json.loads(Path(path).read_text())
    if manifest["split"] == "test":
        raise PermissionError("Test remains sealed")
    records = manifest["records"]
    by_id = {r["content_id"]: {**r, "path": str(require_unsealed_path(dataset_path(r["path"])))} for r in records}
    if set(content_ids) - by_id.keys():
        raise ValueError(f"missing content IDs: {set(content_ids) - by_id.keys()}")
    return [by_id[identity] for identity in content_ids]


@dataclass(frozen=True)
class TemporalWindow:
    frames: torch.Tensor
    valid_frames: int
    offset: int
    frame_indices: list


def temporal_windows(clip, temporal_frames=16, start=0):
    if len(clip) == 0 or temporal_frames < 1:
        raise ValueError("nonempty clip and positive temporal_frames required")
    for offset in range(0, len(clip), temporal_frames):
        real = clip[offset:offset + temporal_frames]
        count = len(real)
        padded = torch.cat([real, real[-1:].expand(temporal_frames - count, -1, -1, -1)])
        yield TemporalWindow(padded, count, offset, list(range(start + offset, start + offset + count)))
