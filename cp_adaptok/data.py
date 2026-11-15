"""Read independent real-video sequences, preserving native resolution outside training crops."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
            for key in ("content_id", "sha256"):
                if value.get(key):
                    identities.append((key, value[key]))
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
    return crop_video(read_video_frames(spec), crop, generator).float() / 255


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


@dataclass(frozen=True)
class TrainingSamplePlan:
    spec: VideoSpec
    crop: tuple | None
    augmentation_seed: int
    horizontal_flip: bool


def plan_training_samples(config, length, crop, streams):
    """Draw data and augmentation choices on the owner thread before CPU prefetch."""
    train, sources = config["training"], config["data"]["train"]
    dimensions = (crop, crop) if isinstance(crop, int) else crop
    dimensions = tuple(dimensions) if dimensions else None
    count = train["gradient_accumulation"] * train.get("batch_sequences", 1)
    plans = []
    for _ in range(count):
        index = int(torch.randint(len(sources), (), generator=streams.data_rng))
        spec = sample_training_spec(sources[index], length, streams.data_rng)
        seed = int(torch.randint(2**63 - 1, (), generator=streams.augmentation_rng))
        plans.append(TrainingSamplePlan(spec, dimensions, seed, train.get("horizontal_flip", False)))
    return plans


def apply_training_transform(video, plan):
    generator = torch.Generator().manual_seed(plan.augmentation_seed)
    h, w = video.shape[-2:]
    ch, cw = plan.crop or (h, w)
    if ch > h or cw > w or min(ch, cw) < 1:
        raise ValueError("training crop exceeds source dimensions; upsampling is forbidden")
    y = int(torch.randint(h - ch + 1, (), generator=generator))
    x = int(torch.randint(w - cw + 1, (), generator=generator))
    flip = plan.horizontal_flip and bool(torch.rand((), generator=generator) < 0.5)
    clip = video[..., y:y + ch, x:x + cw]
    if flip:
        clip = clip.flip(-1)
    transform = {"crop_y": y, "crop_x": x, "crop_height": ch, "crop_width": cw,
                 "horizontal_flip": flip, "augmentation_seed": plan.augmentation_seed}
    return clip.float() / 255, transform


class TrainingClipLoader:
    """One CPU reader owns a bounded LRU cache. No pending plans cross updates."""

    def __init__(self, cache_bytes=256 * 2**20):
        if cache_bytes < 0:
            raise ValueError("cache size must be nonnegative")
        self.limit, self.cached_bytes = cache_bytes, 0
        self.cache = OrderedDict()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpat-video")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.cache.clear()
        self.cached_bytes = 0

    def read(self, plan):
        key = str(resolve(plan.spec.path).resolve()), plan.spec.start, plan.spec.frames
        raw = self.cache.pop(key, None)
        if raw is None:
            raw = read_video_frames(plan.spec)
            size = raw.numel() * raw.element_size()
            self._cache(key, raw, size)
        else:
            self.cache[key] = raw
        clip, transform = apply_training_transform(raw, plan)
        return plan.spec, clip, transform

    def _cache(self, key, raw, size):
        if size > self.limit:
            return
        while self.cached_bytes + size > self.limit:
            _, evicted = self.cache.popitem(last=False)
            self.cached_bytes -= evicted.numel() * evicted.element_size()
        self.cache[key] = raw
        self.cached_bytes += size

    def load(self, plans, prefetch=True):
        if not prefetch:
            for plan in plans:
                yield self.read(plan)
            return
        iterator = iter(plans)
        pending = self.worker.submit(self.read, next(iterator))
        for plan in iterator:
            sample = pending.result()
            pending = self.worker.submit(self.read, plan)
            yield sample
        yield pending.result()


def save_rgb(path, frame):
    from PIL import Image

    array = (
        (frame.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
