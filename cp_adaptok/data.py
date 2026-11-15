"""Read independent real-video sequences, preserving native resolution outside training crops."""

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from .runtime import resolve


@dataclass(frozen=True)
class VideoSpec:
    path: str
    frames: int = 4
    start: int = 0
    fps: float = 30
    purpose: str = "engineering"


def check_splits(config):
    owners = {}
    for split, values in config.items():
        for value in values:
            path = resolve(value["path"]).resolve()
            if path in owners and owners[path] != split:
                raise ValueError(f"train/validation/test leakage: {path}")
            owners[path] = split


def sample_training_spec(spec, length, generator):
    """Sample within the declared frame span using the checkpointed training RNG."""
    spec = VideoSpec(**spec) if isinstance(spec, dict) else spec
    if type(length) is not int or not 1 <= length <= spec.frames:
        raise ValueError("training sequence length must fit the declared source span")
    offset = int(torch.randint(spec.frames - length + 1, (), generator=generator))
    return replace(spec, start=spec.start + offset, frames=length)


def read_video(spec, crop=None, generator=None):
    spec = VideoSpec(**spec) if isinstance(spec, dict) else spec
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
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255)
    finally:
        capture.release()
    video = torch.stack(frames)
    if crop:
        h, w = video.shape[-2:]
        if crop > min(h, w):
            raise ValueError("training crop exceeds source dimensions")
        y = int(torch.randint(h - crop + 1, (), generator=generator))
        x = int(torch.randint(w - crop + 1, (), generator=generator))
        video = video[..., y : y + crop, x : x + crop]
    return video


def save_rgb(path, frame):
    from PIL import Image

    array = (
        (frame.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
