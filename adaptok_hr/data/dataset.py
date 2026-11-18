"""Native pixels only: the existing CP RGB reader plus verified sampling plans."""

from pathlib import Path

import torch

from cp_adaptok.data import VideoSpec, read_video

from ..runtime import file_hash


def verify_selected_frames(row, start, count):
    path = Path(row["path"])
    hashes = row.get("frame_hashes")
    if path.is_dir() and hashes:
        for entry in hashes[start:start + count]:
            if file_hash(path / entry["file"]) != entry["sha256"]:
                raise ValueError("source frame SHA256 mismatch")
    elif path.is_file() and file_hash(path) != row["sha256"]:
        raise ValueError("source container SHA256 mismatch")


def read_clip(row, start, count, height=1080, width=1920, crop_seed=None):
    if start < 0 or count < 1 or start + count > row["original_frames"]:
        raise ValueError("requested frames outside declared source")
    verify_selected_frames(row, start, count)
    clip = read_video(VideoSpec(row["path"], count, start, row["fps"]))
    h, w = clip.shape[-2:]
    if h < 1080 or w < 1920 or row.get("resized"):
        raise ValueError("source is not native high resolution; upsampling forbidden")
    y, x = crop_offset(h, w, height, width, crop_seed)
    clip = clip[..., y:y + height, x:x + width].contiguous()
    return clip, {"content_id": row["content_id"], "source_sha256": row["sha256"],
                  "frame_indices": list(range(start, start + count)), "source_resolution": [w, h],
                  "crop": [x, y, width, height], "resized": False,
                  "engineering_crop": (height, width) != (1080, 1920), "fps": row["fps"],
                  "color_format": "RGB float32 [0,1]; OpenCV BGR->RGB"}


def crop_offset(h, w, height, width, seed):
    if h < height or w < width:
        raise ValueError("crop exceeds source resolution")
    if seed is None:
        return (h - height) // 2, (w - width) // 2
    rng = torch.Generator().manual_seed(seed)
    return int(torch.randint(h - height + 1, (), generator=rng)), int(torch.randint(w - width + 1, (), generator=rng))


class ClipSampler:
    def __init__(self, records, seed):
        if not records:
            raise ValueError("no native high-resolution training sources")
        self.records = records
        self.rng = torch.Generator().manual_seed(seed)
        self.plans = 0

    def plan(self, length):
        eligible = [i for i, r in enumerate(self.records) if r["original_frames"] >= length]
        if not eligible:
            raise ValueError("no source can supply the requested sequence length")
        index = eligible[int(torch.randint(len(eligible), (), generator=self.rng))]
        row = self.records[index]
        start = int(torch.randint(row["original_frames"] - length + 1, (), generator=self.rng))
        crop_seed = int(torch.randint(2**62, (), generator=self.rng))
        self.plans += 1
        return {"record": index, "start": start, "length": length, "crop_seed": crop_seed}

    def sample(self, length, height, width):
        plan = self.plan(length)
        clip, metadata = read_clip(self.records[plan["record"]], plan["start"], plan["length"],
                                   height, width, plan["crop_seed"])
        return clip, {**metadata, "plan": plan}

    def state_dict(self):
        return {"rng": self.rng.get_state(), "plans": self.plans}

    def load_state_dict(self, state):
        self.rng.set_state(state["rng"])
        self.plans = state["plans"]
