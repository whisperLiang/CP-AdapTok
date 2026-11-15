"""Import original DAVIS train sequences, keeping official validation/test separate."""

import argparse
import binascii
import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image

from cp_adaptok.runtime import resolve, write_json

ARCHIVE_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip"


def official_train_names(archive):
    train = archive.read("DAVIS/ImageSets/2017/train.txt").decode().splitlines()
    validation = set(archive.read("DAVIS/ImageSets/2017/val.txt").decode().splitlines())
    if set(train) & validation:
        raise ValueError("official DAVIS split overlap")
    return train


def native_sequence(archive, output, name, frames):
    prefix = f"DAVIS/JPEGImages/Full-Resolution/{name}/"
    paths = sorted(p for p in archive.namelist() if p.startswith(prefix) and p.endswith(".jpg"))
    if len(paths) < frames:
        return None
    first = output / "train" / name / Path(paths[0]).name
    source = first if first.exists() else io.BytesIO(archive.read(paths[0]))
    with Image.open(source) as image:
        width, height = image.size
    if height < 1080 or width < 1920:
        return None
    return paths[:frames], [width, height]


def verified_frame(archive, path, destination):
    data = destination.read_bytes() if destination.exists() else archive.read(path)
    info = archive.getinfo(path)
    if len(data) != info.file_size or binascii.crc32(data) != info.CRC:
        raise ValueError(f"archive CRC differs: {destination}")
    if not destination.exists():
        destination.write_bytes(data)
    return data


def extract_sequence(archive, output, name, paths, size):
    digest = hashlib.sha256()
    for path in paths:
        destination = output / "train" / name / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest.update(verified_frame(archive, path, destination))
    print(f"extracted {name}: {size[0]}x{size[1]}", flush=True)
    return {
        "path": str((output / "train" / name).relative_to(resolve("."))),
        "frames": len(paths),
        "fps": 30,
        "purpose": "official-DAVIS2017-train-native-pixels-nominal-fps",
        "content_id": f"DAVIS2017:{name}",
        "source_size": size,
        "sha256": digest.hexdigest(),
        "official_split": "train",
    }


def import_sequences(archive, output, sequences, frames):
    records = []
    with zipfile.ZipFile(archive) as z:
        for name in official_train_names(z):
            sequence = native_sequence(z, output, name, frames)
            if sequence is None:
                continue
            paths, size = sequence
            records.append(extract_sequence(z, output, name, paths, size))
            if len(records) == sequences:
                break
    if not records:
        raise ValueError("no native >=1080p official training sequences")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/davis2017/trainval-full.zip")
    parser.add_argument("--output", default="data/davis2017")
    parser.add_argument("--sequences", type=int, default=12)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--proxy")
    args = parser.parse_args()
    archive, output = resolve(args.archive), resolve(args.output)
    if args.remote:
        from scripts.remote_archive import RangeReader

        archive = RangeReader(ARCHIVE_URL, args.proxy)
    records = import_sequences(archive, output, args.sequences, args.frames)
    manifest = {
        "source": "https://davischallenge.org/davis2017/code.html",
        "archive_url": ARCHIVE_URL,
        "archive_sha256": None
        if args.remote
        else hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(),
        "download_mode": "selected ZIP entries via verified HTTP ranges and ZIP CRC"
        if args.remote
        else "full ZIP",
        "official_train_only": True,
        "resized": False,
        "fps_note": "nominal 30; original image sequences have no per-frame timestamp metadata",
        "records": records,
        "training_specs": [
            {k: v for k, v in r.items() if k in ("path", "frames", "fps", "purpose", "content_id")}
            for r in records
        ],
    }
    write_json(output / "manifest.json", manifest)
    print([(r["content_id"], r["source_size"]) for r in records])


if __name__ == "__main__":
    main()
