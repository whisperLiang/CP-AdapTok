"""Reuse CP manifests, RGB decoding, split checks and original frame hashes."""

from concurrent.futures import ThreadPoolExecutor

from aip_vtok.runtime import file_hash, resolve
from cp_adaptok.data import VideoSpec, read_video_frames
from cp_adaptok.large_scale_data import load_manifests


def sources(config, split):
    if split not in ("train", "validation"):
        raise ValueError("Gate 0 cannot open sealed test frames")
    splits, hashes = load_manifests(config["manifest_dir"])
    rows = [r for r in splits[split] if r["resolution"] == [1920, 1080] and not r.get("resized", False)]
    return rows, hashes


def source_by_id(rows, content_id):
    return next(row for row in rows if row["content_id"] == content_id)


def read_window(row, start, length):
    if start < 0 or start + length > row["original_frames"]:
        raise ValueError("window exceeds manifest span; repeating tail frames is forbidden")
    frames = row["frame_hashes"][start : start + length]
    for item in frames:
        if file_hash(resolve(row["path"]) / item["file"]) != item["sha256"]:
            raise ValueError("source frame differs from audited manifest")
    return read_video_frames(VideoSpec(row["path"], frames=length, start=start))


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
