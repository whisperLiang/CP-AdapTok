"""Pinned clip plans, shared Validation loader, and export-safe clip selection."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torch import Tensor

from common.data import AuditedFrames, load_cohort
from common.paths import ROOT
from common.video import NATIVE_SHAPE

COHORT = ROOT / "configs/validation_cohort.json"
PLAN = ROOT / "configs/sequence_manifest.json"


def json_at(path):
    return json.loads(path.read_bytes())


def setup(seed=1):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

def frozen_data():
    identity = json_at(COHORT)
    reader = AuditedFrames(ROOT / "data/manifests", identity["manifest_hashes"])
    return reader, load_cohort(COHORT, reader), identity


@dataclass(frozen=True)
class Clip:
    split: str
    video: str
    start: int
    length: int

    def manifest(self, reader: AuditedFrames) -> dict:
        row = reader.catalog(self.split)[self.video]
        selection = row["frame_hashes"][self.start:self.start + self.length]
        indexes = [int(Path(item["file"]).stem) for item in selection]
        if len(indexes) != self.length or indexes != list(range(indexes[0], indexes[0] + self.length)):
            raise ValueError("temporal clip has a frame gap or crosses the source boundary")
        if row["resized"] or row["resolution"] != [1920, 1080]:
            raise ValueError("temporal clip must be native 1080p")
        return vars(self) | {"fps": row["fps"], "fps_nominal": row["fps_nominal"], "frames": [
            {"index": self.start + i, "source_index": index, "timestamp_seconds_nominal": index / row["fps"],
             "sha256": selection[i]["sha256"]} for i, index in enumerate(indexes)
        ]}

def source_frame(reader: AuditedFrames, clip: Clip, offset: int) -> torch.Tensor:
    return transfer(reader.read(clip.split, clip.video, clip.start + offset)[None])


def transfer(rgb):
    return rgb.cuda(non_blocking=True).to(dtype=torch.float32, memory_format=torch.channels_last).div_(255)


@dataclass(frozen=True)
class Source:
    clip: Clip
    fps: float
    rgb: Tensor

def source_clips(reader: AuditedFrames, clips: list[Clip]) -> list[Source]:
    result = []
    for clip in clips:
        spec = reader.catalog(clip.split)[clip.video]
        if spec["resized"] or tuple(reversed(spec["resolution"])) != NATIVE_SHAPE:
            raise ValueError(f"{clip.video} must be native 1080p")
        frames = torch.stack([reader.read(clip.split, clip.video, clip.start + i) for i in range(clip.length)])
        rgb = frames.cuda().float().div_(255).contiguous(memory_format=torch.channels_last)
        result.append(Source(clip, spec["fps"], rgb))
    return result


def _check(reader: AuditedFrames, rows: list[dict]) -> list[Clip]:
    clips = [Clip(row["split"], row["video"], row["start"], row["length"]) for row in rows]
    if any(clip.manifest(reader) != row for clip, row in zip(clips, rows, strict=True)):
        raise ValueError("archived clip identity differs from the frozen manifest")
    return clips

def inputs() -> tuple[AuditedFrames, dict[str, list[Clip]]]:
    reader, cohort, _ = frozen_data()
    archive = json_at(PLAN)
    if archive["dataset_sha256"] != reader.hashes:
        raise ValueError("frozen temporal manifest hashes changed")
    primary = _check(reader, archive["validation"])
    if {clip.video: clip.length for clip in primary} != cohort:
        raise ValueError("Validation cohort no longer matches the 174-frame archive")
    fit, holdout = (_check(reader, [row for row in archive["training"][name] if row["length"] == 32])
                    for name in ("fit", "holdout"))
    if any(clip.split != "train" for clip in (*fit, *holdout)):
        raise PermissionError("fitting and calibration must stay inside Train")
    if {clip.video for clip in fit} & {clip.video for clip in holdout}:
        raise ValueError("Calibration overlaps fitting sources")
    return reader, {"validation": primary, "long": _check(reader, archive["long_validation"]),
                    "fit": fit, "calibration": holdout}
