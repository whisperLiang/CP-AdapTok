"""Auditable native-pixel datasets. Manifest metadata never enters the model graph."""

import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image

from cp_adaptok.data import VideoSpec, check_splits
from cp_adaptok.runtime import resolve, write_json
from cp_adaptok.training_state import stable_hash

CROPS = ((512, 512), (720, 1280), (1080, 1920))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FORBIDDEN_TRAIN = ("uvg", "mcl_jcv", "mcl-jcv", "basketball", "historical-development")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_sequence_metadata(path):
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"empty image sequence: {path}")
    sizes, digest, frame_hashes = set(), hashlib.sha256(), []
    for frame in paths:
        with Image.open(frame) as image:
            sizes.add(image.size)
            image.verify()
        sha = file_hash(frame)
        frame_hashes.append({"file": frame.name, "sha256": sha})
        digest.update(bytes.fromhex(sha))
    if len(sizes) != 1:
        raise ValueError(f"variable frame resolution: {path}")
    validate_image_timeline(paths, path)
    return {"resolution": list(sizes.pop()), "original_frames": len(paths),
            "sha256": digest.hexdigest(), "frame_hashes": frame_hashes}


def validate_image_timeline(paths, path):
    numeric = [int(p.stem) for p in paths if p.stem.isdigit()]
    if len(numeric) == len(paths) and numeric != list(range(numeric[0], numeric[0] + len(paths))):
        raise ValueError(f"nonconsecutive image sequence: {path}")


def container_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    count, dimensions = 0, set()
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            dimensions.add((frame.shape[1], frame.shape[0]))
    finally:
        capture.release()
    if count < 1 or len(dimensions) != 1:
        raise ValueError(f"empty or variable-resolution video: {path}")
    return {"resolution": list(dimensions.pop()), "original_frames": count,
            "container_fps": fps, "sha256": file_hash(path)}


def audit_source(path, content_id, source, official_split, fps=30, fps_nominal=True, role="train"):
    path = resolve(path).resolve()
    details = image_sequence_metadata(path) if path.is_dir() else container_metadata(path)
    width, height = details["resolution"]
    return {"path": str(path), "content_id": content_id, "source": source, "official_split": official_split,
            "role": role, "fps": details.get("container_fps", fps), "fps_nominal": fps_nominal,
            "fps_note": "nominal: no image timestamp metadata" if fps_nominal else "container metadata",
            "supported_crops": [list(crop) for crop in CROPS if height >= crop[0] and width >= crop[1]],
            "native_1080p_eligible": height >= 1080 and width >= 1920, "resized": False, **details}


def validate_records(splits):
    check_splits(splits)
    for row in splits.get("train", []):
        validate_training_source(row)
    for rows in splits.values():
        for row in rows:
            if row["original_frames"] < 4 or row.get("resized"):
                raise ValueError(f"invalid native video: {row['content_id']}")


def validate_training_source(row):
    text = " ".join(str(row.get(k, "")) for k in ("source", "path", "content_id", "role")).lower()
    if row["official_split"] != "train" or row.get("role") != "train":
        raise ValueError(f"training requires an explicit official train split: {row['content_id']}")
    if any(name in text for name in FORBIDDEN_TRAIN):
        raise ValueError(f"historical test content is forbidden for training: {row['content_id']}")


def summarize_split(rows):
    frames = sum(row["original_frames"] for row in rows)
    high = sum(row["original_frames"] for row in rows if row["native_1080p_eligible"])
    return {"independent_videos": len({row["content_id"] for row in rows}), "frames": frames,
            "native_1080p_frames": high, "native_1080p_frame_fraction": high / frames if frames else 0,
            "eligible_videos_by_crop": {f"{h}x{w}": sum([h, w] in r["supported_crops"] for r in rows)
                                         for h, w in CROPS}}


def write_manifests(splits, provenance, directory="data/manifests"):
    validate_records(splits)
    hashes, summaries = {}, {}
    for split, rows in splits.items():
        manifest = {"schema_version": 1, "split": split, "provenance": provenance, "records": rows}
        write_json(resolve(directory) / f"cpat_{split}.json", manifest)
        hashes[split] = stable_hash(manifest)
        summaries[split] = summarize_split(rows)
    train = summaries["train"]
    sufficient = train["independent_videos"] >= 100 and train["frames"] >= 10000
    audit = {"splits": summaries, "manifest_hashes": hashes, "content_leakage": False,
             "planning_targets": {"independent_train_videos": 100, "train_frames": 10000},
             "scale_ready": sufficient, "convergence_verified": False, "provenance": provenance}
    write_json("results/large_scale/dataset_audit.json", audit)
    return audit


def load_manifests(directory="data/manifests"):
    manifests = {split: json.loads((resolve(directory) / f"cpat_{split}.json").read_text())
                 for split in ("train", "validation", "test")}
    splits = {split: value["records"] for split, value in manifests.items()}
    validate_records(splits)
    return splits, {split: stable_hash(value) for split, value in manifests.items()}


def video_specs(rows, crop=None, frames=None):
    result = []
    dimensions = [crop, crop] if isinstance(crop, int) else crop
    for row in rows:
        width, height = row["resolution"]
        if dimensions and (height < dimensions[0] or width < dimensions[1]):
            continue
        length = row["original_frames"] if frames is None else min(frames, row["original_frames"])
        result.append(vars(VideoSpec(row["path"], length, fps=row["fps"], purpose=row["role"],
                                     content_id=row["content_id"])))
    return result
