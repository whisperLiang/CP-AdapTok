"""Reuse CP manifests, RGB decoding, split checks and original frame hashes."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from aip_vtok.runtime import resolve
from cp_adaptok.data import rgb_tensor
from cp_adaptok.large_scale_data import load_manifests


def sources(config, split):
    if split not in ("train", "validation"):
        raise ValueError("Gate 0 cannot open sealed test frames")
    splits, hashes = load_manifests(config["manifest_dir"])
    rows = [r for r in splits[split] if r["resolution"] == [1920, 1080] and not r.get("resized", False)]
    return rows, hashes


def source_by_id(rows, content_id):
    return next(row for row in rows if row["content_id"] == content_id)


def verified_rgb(directory, item):
    root = resolve(directory).resolve()
    path = (root / item["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest frame escapes its declared source")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("source frame differs from audited manifest")
    return rgb_tensor(cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR))


def read_window(row, start, length):
    if start < 0 or start + length > row["original_frames"]:
        raise ValueError("window exceeds manifest span; repeating tail frames is forbidden")
    frames = row["frame_hashes"][start : start + length]
    if len(frames) != length:
        raise ValueError("manifest is missing frame hashes")
    clip = torch.stack([verified_rgb(row["path"], item) for item in frames])
    width, height = row["resolution"]
    if clip.shape[-2:] != (height, width):
        raise ValueError("decoded resolution differs from manifest")
    return clip


def training_plans(rows, steps, length, rng):
    import torch

    plans = []
    for _ in range(steps):
        index = int(torch.randint(len(rows), (), generator=rng.data_rng))
        start = int(torch.randint(rows[index]["original_frames"] - length + 1, (), generator=rng.data_rng))
        flip = bool(torch.randint(2, (), generator=rng.aug_rng))
        plans.append({"source": index, "start": start, "length": length, "flip": flip})
    return plans


def prefetched_windows(rows, plans):
    """Exactly one pending read. Plans live in the checkpoint, not in a worker RNG."""

    def read(plan):
        return read_window(rows[plan["source"]], plan["start"], plan["length"])

    with ThreadPoolExecutor(max_workers=1) as pool:
        iterator = iter(plans)
        first = next(iterator, None)
        if first is None:
            return
        pending = pool.submit(read, first)
        for plan in iterator:
            clip = pending.result()
            pending = pool.submit(read, plan)
            yield clip
        yield pending.result()
