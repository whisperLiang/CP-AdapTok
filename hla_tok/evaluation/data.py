"""Validation and training readers bound to the frozen CP manifests."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from hla_tok.progressive.geometry import NATIVE_SHAPE

ALLOWED_ROLES = {"train": {"train"}, "validation": {"validation", "historical-development"}}


def load_manifests(directory: Path) -> tuple[dict, dict]:
    records, hashes = {}, {}
    for split in ("train", "validation", "test"):
        manifest = json.loads((directory / f"cpat_{split}.json").read_bytes())
        if manifest["split"] != split:
            raise ValueError("manifest split differs from the requested split")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        hashes[split] = hashlib.sha256(canonical).hexdigest()
        records[split] = manifest["records"]
    return records, hashes


class AuditedFrames:
    def __init__(self, manifest_dir: Path, expected_hashes: dict[str, str]):
        splits, hashes = load_manifests(manifest_dir.resolve())
        if hashes != expected_hashes:
            raise ValueError("manifest identity differs from the frozen Joint V1 provenance")
        self.hashes = hashes
        self.counts = dict.fromkeys(splits, 0)
        self._sealed_paths = tuple(Path(row["path"]).resolve() for row in splits["test"])
        self._sealed_hashes = {frame["sha256"] for row in splits["test"] for frame in row["frame_hashes"]}
        self._sealed_videos = {row["content_id"] for row in splits["test"]}
        self._catalog = {split: {row["content_id"]: row for row in rows} for split, rows in splits.items()
                         if split in ALLOWED_ROLES}
        self._check_split_isolation(splits)

    @staticmethod
    def _check_split_isolation(splits: dict) -> None:
        owners: dict[str, str] = {}
        for split, rows in splits.items():
            for row in rows:
                for frame in row["frame_hashes"]:
                    if owners.setdefault(frame["sha256"], split) != split:
                        raise ValueError("a frame digest belongs to more than one split")

    def catalog(self, split: str) -> dict:
        if split not in ALLOWED_ROLES:
            raise PermissionError("Test remains sealed")
        return self._catalog[split]

    def historical_intersection(self, historical_sources: list[dict]) -> dict:
        ids = {row["content_id"] for row in historical_sources}
        hashes = {item["sha256"] for row in historical_sources for item in row.get("frame_hashes", [])}
        return {
            "test_videos": sorted(ids & self._sealed_videos),
            "test_source_file_hashes_in_historical_manifest": len(hashes & self._sealed_hashes),
            "validation_videos": sorted(ids & self.catalog("validation").keys()),
            "train_videos": sorted(ids & self.catalog("train").keys()),
        }

    def _open_allowed(self, row: dict, item: dict) -> bytes:
        root = Path(row["path"]).resolve()
        path = (root / item["file"]).resolve()
        if any(path.is_relative_to(sealed) for sealed in self._sealed_paths) or item["sha256"] in self._sealed_hashes:
            raise PermissionError("manifest entry aliases sealed Test content")
        if not path.is_relative_to(root):
            raise ValueError("source frame escapes its manifest directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"source bytes do not match audited manifest: {path}")
        return data

    def read(self, split: str, video: str, index: int) -> Tensor:
        row = self.catalog(split)[video]
        if row["role"] not in ALLOWED_ROLES[split] or video in self._sealed_videos:
            raise PermissionError("source role is outside the allowed split")
        if type(index) is not int or not 0 <= index < len(row["frame_hashes"]):
            raise ValueError("frame index is outside the manifest")
        data = self._open_allowed(row, row["frame_hashes"][index])
        self.counts[split] += 1
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or tuple(image.shape[:2]) != tuple(reversed(row["resolution"])):
            raise ValueError(f"decoded RGB dimensions differ from the manifest: {video} frame {index}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def load_cohort(path: Path, reader: AuditedFrames) -> dict[str, int]:
    """Cross-check every archived Joint V1 frame against the frozen manifest before opening pixels."""
    identity = json.loads(path.read_text())
    if identity["manifest_hashes"] != reader.hashes:
        raise ValueError("cohort and dataset manifest identities differ")
    counts: Counter[str] = Counter()
    for row in identity["frames"]:
        video, index = row["video"], row["index"]
        source = reader.catalog("validation")[video]
        if tuple(reversed(source["resolution"])) != NATIVE_SHAPE or source["resized"]:
            raise ValueError("matched Validation frame is not native 1080p")
        if index != counts[video] or source["frame_hashes"][index]["sha256"] != row["source_sha256"]:
            raise ValueError("matched Validation frame identity differs")
        counts[video] += 1
    if sum(counts.values()) != 174:
        raise ValueError("matched Validation requires exactly 174 source frames")
    return dict(counts)
