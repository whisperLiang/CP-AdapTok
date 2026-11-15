"""Extract and audit complete official DAVIS splits, keeping prior development content explicit."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from cp_adaptok.large_scale_data import audit_source, file_hash, write_manifests
from cp_adaptok.runtime import resolve
from scripts.prepare_optimization_data import ARCHIVE_URL, official_train_names, verified_frame

SOURCE = "https://davischallenge.org/davis2017/code.html"


def extract_images(archive, name, output):
    prefix = f"DAVIS/JPEGImages/Full-Resolution/{name}/"
    entries = sorted(p for p in archive.namelist() if p.startswith(prefix) and p.endswith(".jpg"))
    if not entries:
        raise ValueError(f"missing official sequence: {name}")
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        verified_frame(archive, entry, destination / Path(entry).name)
    return destination


def historical_davis_names():
    names = set()
    for path in resolve("results/prefix_utilization/data").rglob("manifest.json"):
        for row in json.loads(path.read_text()).get("records", []):
            identity = row.get("content_id", "")
            if identity.startswith("DAVIS2017:"):
                names.add(identity.split(":", 1)[1])
    return names


def prepare(archive_path, output):
    splits = {"train": [], "validation": [], "test": []}
    historical = historical_davis_names()
    with ZipFile(archive_path) as archive:
        train = official_train_names(archive)
        validation = archive.read("DAVIS/ImageSets/2017/val.txt").decode().split()
        # Freeze the final holdout before any new model selection. Historical val stays development-only.
        untouched = sorted(set(validation) - historical)
        final = set(untouched[::3])
        if not final:
            raise ValueError("no untouched final holdout; supply a new lawful independent dataset")
        for name in train + validation:
            split = "train" if name in train else "test" if name in final else "validation"
            role = "train" if split == "train" else "historical-development" if name in historical else split
            path = extract_images(archive, name, output / split)
            row = audit_source(path, f"DAVIS2017:{name}", SOURCE, "train" if name in train else "val", role=role)
            splits[split].append(row)
            print(f"{split}: {name}: {row['original_frames']} frames {row['resolution']}", flush=True)
    provenance = {"source": SOURCE, "archive_url": ARCHIVE_URL, "archive_sha256": file_hash(archive_path),
                  "zip_crc_verified": True, "complete_sequences": True, "fps_nominal": True,
                  "historical_development_content_ids": [f"DAVIS2017:{name}" for name in sorted(historical)],
                  "excluded_legacy_test": ["MCL-JCV", "UVG", "BasketballDrive"],
                  "final_holdout_rule": "sorted untouched official val, every third sequence; frozen before training"}
    return write_manifests(splits, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/davis2017/DAVIS-2017-trainval-Full-Resolution.zip.part")
    parser.add_argument("--output", default="data/large_scale/davis2017")
    args = parser.parse_args()
    print(json.dumps(prepare(resolve(args.archive), resolve(args.output)), indent=2))


if __name__ == "__main__":
    main()
