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
    content_id: str | None = None


def check_splits(config):
    owners = {}
    for split, values in config.items():
        for value in values:
            identities = [resolve(value["path"]).resolve()]
            if value.get("content_id"):
                identities.append(value["content_id"])
            for identity in identities:
                if identity in owners and owners[identity] != split:
                    raise ValueError(f"train/validation/test leakage: {identity}")
                owners[identity] = split


def sample_training_spec(spec, length, generator):
    """Sample within the declared frame span using the checkpointed training RNG."""
    spec = VideoSpec(**spec) if isinstance(spec, dict) else spec
    if type(length) is not int or not 1 <= length <= spec.frames:
        raise ValueError("training sequence length must fit the declared source span")
    offset = int(torch.randint(spec.frames - length + 1, (), generator=generator))
    return replace(spec, start=spec.start + offset, frames=length)


def read_video(spec, crop=None, generator=None):
    spec = VideoSpec(**spec) if isinstance(spec, dict) else spec
    source = resolve(spec.path)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        selected = paths[spec.start : spec.start + spec.frames]
        if len(selected) != spec.frames:
            raise ValueError(f"not enough frames: {spec.path}")
        frames = [rgb_tensor(cv2.imread(str(path))) for path in selected]
    else:
        frames = read_container(spec)
    return crop_video(torch.stack(frames), crop, generator).float() / 255


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


def crop_video(video, crop, generator):
    if crop:
        h, w = video.shape[-2:]
        ch, cw = (crop, crop) if isinstance(crop, int) else crop
        if ch > h or cw > w or min(ch, cw) < 1:
            raise ValueError("training crop exceeds source dimensions")
        y = int(torch.randint(h - ch + 1, (), generator=generator))
        x = int(torch.randint(w - cw + 1, (), generator=generator))
        video = video[..., y : y + ch, x : x + cw]
    return video


def save_rgb(path, frame):
    from PIL import Image

    array = (
        (frame.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
