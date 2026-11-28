"""Native RGB decoding and geometry shared by all evaluations."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from common.paths import ROOT
from third_party.DCVC.src.utils.transforms import ycbcr2rgb

NATIVE_SHAPE = (1080, 1920)


def read_video(spec):
    """Decode exactly the requested contiguous window as RGB float32 in [0, 1]."""
    return read_video_frames(spec).float() / 255


def resolve(path):
    return ROOT / Path(path).expanduser()


@dataclass(frozen=True)
class VideoSpec:
    path: str
    frames: int = 4
    start: int = 0
    fps: float = 30
    purpose: str = "engineering"
    content_id: str | None = None

def read_video_frames(spec):
    """Decode exactly the requested contiguous window, retaining uint8 CPU pixels."""
    source = resolve(spec.path)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        selected = paths[spec.start : spec.start + spec.frames]
        if len(selected) != spec.frames:
            raise ValueError(f"not enough frames: {spec.path}")
        frames = [rgb_tensor(cv2.imread(str(path))) for path in selected]
    else:
        frames = read_container(spec)
    return torch.stack(frames)

def rgb_tensor(bgr):
    if bgr is None:
        raise ValueError("unreadable video frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

def read_container(spec):
    capture = cv2.VideoCapture(str(resolve(spec.path)))
    if not capture.isOpened():
        raise ValueError(f"cannot open video {spec.path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, spec.start)
    frames = []
    try:
        for _ in range(spec.frames):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"not enough frames: {spec.path}")
            frames.append(rgb_tensor(bgr))
    finally:
        capture.release()
    return frames


def restore_rgb(centered_yuv, height, width):
    return ycbcr2rgb((centered_yuv[..., :height, :width] + 0.5).clamp(0, 1)).clamp(0, 1)
