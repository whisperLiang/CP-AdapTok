"""Extract untouched official DAVIS validation sequences with CRC and split evidence."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from cp_adaptok.runtime import resolve, write_json
from scripts.prepare_optimization_data import ARCHIVE_URL, verified_frame
from scripts.remote_archive import RangeReader


def extract_heldout(archive, output, sequences, frames):
    train = set(archive.read("DAVIS/ImageSets/2017/train.txt").decode().splitlines())
    validation = archive.read("DAVIS/ImageSets/2017/val.txt").decode().splitlines()
    if train.intersection(validation):
        raise ValueError("official splits overlap")
    records = []
    for name in validation:
        prefix = f"DAVIS/JPEGImages/Full-Resolution/{name}/"
        paths = sorted(p for p in archive.namelist() if p.startswith(prefix) and p.endswith(".jpg"))[:frames]
        if len(paths) < frames:
            continue
        with Image.open(io.BytesIO(archive.read(paths[0]))) as image:
            size = image.size
        if size != (1920, 1080):
            continue
        digest = hashlib.sha256()
        for path in paths:
            destination = output / name / Path(path).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest.update(verified_frame(archive, path, destination))
        records.append({"path": str(output / name), "frames": frames, "fps": 30,
                        "purpose": "heldout-official-DAVIS-val-native1080-nominal-fps",
                        "content_id": f"DAVIS2017:{name}", "sha256": digest.hexdigest(),
                        "size": list(size), "official_split": "val"})
        print(f"heldout {name}: {frames} native 1080p frames", flush=True)
        if len(records) == sequences:
            break
    if len(records) != sequences:
        raise ValueError("not enough native 1080p independent validation sequences")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/prefix_utilization/data/heldout1080")
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    output = resolve(args.output)
    reader = RangeReader(ARCHIVE_URL, args.proxy)
    with ZipFile(reader) as archive:
        records = extract_heldout(archive, output, args.sequences, args.frames)
    keys = ("path", "frames", "fps", "purpose", "content_id")
    write_json(output / "manifest.json", {"archive_url": ARCHIVE_URL, "resized": False,
               "downloaded_bytes": reader.downloaded, "crc_verified": True, "records": records,
               "specs": [{k: row[k] for k in keys} for row in records]})


if __name__ == "__main__":
    main()
