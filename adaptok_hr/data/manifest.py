"""Separate HR index built from existing CP manifests or explicitly licensed local sources."""

import argparse
import json
from pathlib import Path

from cp_adaptok.data import check_splits

from ..runtime import file_hash, stable_hash, write_json

HISTORICAL = {"DAVIS2017:cows", "DAVIS2017:goat", "DAVIS2017:parkour"}


def is_native(row):
    w, h = row["resolution"]
    return w >= 1920 and h >= 1080 and not row.get("resized", False)


def load_external(path):
    data = json.loads(Path(path).read_text())
    if not data.get("usage_conditions") or not data.get("license"):
        raise ValueError("external sources require explicit license and usage_conditions; no downloads")
    rows = data["records"]
    for row in rows:
        row.setdefault("usage_conditions", data["usage_conditions"])
        row.setdefault("license", data["license"])
    return rows


def validate_row(row, split):
    if not is_native(row):
        raise ValueError(f"native resolution below 1920x1080 or resized: {row['content_id']}")
    if row["original_frames"] < 1 or not row.get("sha256"):
        raise ValueError("source frame count and SHA256 required")
    if split == "train" and (row["official_split"] != "train" or row.get("role") != "train"):
        raise ValueError("training requires an explicit train source")
    if split == "train" and row["content_id"] in HISTORICAL:
        raise ValueError("historical test source cannot train")


def validate_index(index):
    splits = index["splits"]
    check_splits(splits)
    for split, rows in splits.items():
        for row in rows:
            validate_row(row, split)


def partition_holdout(splits):
    historical = [r for r in splits["test"] if r["content_id"] in HISTORICAL]
    splits["test"] = [r for r in splits["test"] if r["content_id"] not in HISTORICAL]
    splits["historical"] = historical


def append_external(splits, external):
    for path in external:
        for row in load_external(path):
            splits[row["role"]].append(row)


def summarize_splits(splits):
    return {s: {"sources": len(rows), "frames": sum(r["original_frames"] for r in rows)}
            for s, rows in splits.items()}


def build_index(directory, external=()):
    paths = {s: Path(directory) / f"cpat_{s}.json" for s in ("train", "validation", "test")}
    splits = {s: [r for r in json.loads(p.read_text())["records"] if is_native(r)] for s, p in paths.items()}
    partition_holdout(splits)
    append_external(splits, external)
    index = {"schema_version": 1, "splits": splits,
             "cp_manifest_sha256": {s: file_hash(p) for s, p in paths.items()},
             "external_sha256": {str(p): file_hash(p) for p in external},
             "color_format": "OpenCV decoded RGB float32/255; no resize",
             "test_policy": "HR holdout excludes three historically evaluated exact-1080p videos; old CP-indexed sources, not a new independent dataset",
             "summary": summarize_splits(splits)}
    validate_index(index)
    index["identity"] = stable_hash(index)
    return index


def load_index(path):
    index = json.loads(Path(path).read_text())
    identity = index.pop("identity")
    if stable_hash(index) != identity:
        raise ValueError("HR manifest identity mismatch")
    index["identity"] = identity
    validate_index(index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-manifests", default="data/manifests")
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--output", default="data/adaptok_hr/index.json")
    args = parser.parse_args()
    index = build_index(args.cp_manifests, args.external)
    write_json(args.output, index)
    write_json("results/adaptok_hr/data_audit.json", {k: v for k, v in index.items() if k != "splits"})
    print(json.dumps(index["summary"]))


if __name__ == "__main__":
    main()
