"""Manifest-owned splits and native pixels, with no resize or test-data sampling."""

from pathlib import Path

from cp_adaptok.data import VideoSpec, read_video
from cp_adaptok.large_scale_data import file_hash, load_manifests


def records(config, split, videos=None, exact=True):
    splits, hashes = load_manifests(config["manifest_dir"])
    selected = native_records(splits[split]) if exact else splits[split]
    if videos:
        selected = [row for row in selected if row["content_id"] in videos]
        if set(videos) != {row["content_id"] for row in selected}:
            raise ValueError("requested videos are missing or have incompatible native resolution")
    if not selected:
        raise ValueError("empty data selection")
    return selected, hashes


def native_records(rows):
    return [row for row in rows if row["resolution"] == [1920, 1080]]


def read_clip(row, length, start=0, crop=None):
    if start < 0 or start + length > row["original_frames"]:
        raise ValueError("frame window exceeds the audited source")
    for frame in row["frame_hashes"][start : start + length]:
        if file_hash(Path(row["path"]) / frame["file"]) != frame["sha256"]:
            raise ValueError("input frame differs from the frozen manifest")
    clip = read_video(VideoSpec(row["path"], frames=length, start=start, content_id=row["content_id"]))
    if crop:
        h, w = clip.shape[-2:]
        ch, cw = crop
        if h < ch or w < cw:
            raise ValueError("native training input is too small; upsampling is forbidden")
        clip = clip[..., (h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw]
    if clip.shape[-2:] != (1080, 1920):
        raise ValueError("this experiment requires actual native 1920x1080 frames")
    return clip
