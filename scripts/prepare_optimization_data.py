"""Import original DAVIS train sequences, keeping official validation/test separate."""

import argparse
import binascii
import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image

from cp_adaptok.runtime import resolve, write_json


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
    records = []
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip"
    if args.remote:
        from scripts.remote_archive import RangeReader

        archive = RangeReader(url, args.proxy)
    with zipfile.ZipFile(archive) as z:
        split_path = "DAVIS/ImageSets/2017/train.txt"
        train = z.read(split_path).decode().splitlines()
        validation = set(z.read("DAVIS/ImageSets/2017/val.txt").decode().splitlines())
        if set(train) & validation:
            raise ValueError("official DAVIS split overlap")
        for name in train:
            prefix = f"DAVIS/JPEGImages/Full-Resolution/{name}/"
            paths = sorted(p for p in z.namelist() if p.startswith(prefix) and p.endswith(".jpg"))
            if len(paths) < args.frames:
                continue
            first = output / "train" / name / Path(paths[0]).name
            image = Image.open(first) if first.exists() else Image.open(io.BytesIO(z.read(paths[0])))
            width, height = image.size
            if height < 1080 or width < 1920:
                continue
            digest = hashlib.sha256()
            for path in paths[: args.frames]:
                destination = output / "train" / name / Path(path).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = destination.read_bytes() if destination.exists() else z.read(path)
                info = z.getinfo(path)
                if len(data) != info.file_size or binascii.crc32(data) != info.CRC:
                    raise ValueError(f"archive CRC differs: {destination}")
                digest.update(data)
                if not destination.exists():
                    destination.write_bytes(data)
            records.append(
                {
                    "path": str((output / "train" / name).relative_to(resolve("."))),
                    "frames": args.frames,
                    "fps": 30,
                    "purpose": "official-DAVIS2017-train-native-pixels-nominal-fps",
                    "content_id": f"DAVIS2017:{name}",
                    "source_size": [width, height],
                    "sha256": digest.hexdigest(),
                    "official_split": "train",
                }
            )
            print(f"extracted {name}: {width}x{height}", flush=True)
            if len(records) == args.sequences:
                break
    if not records:
        raise ValueError("no native >=1080p official training sequences")
    manifest = {
        "source": "https://davischallenge.org/davis2017/code.html",
        "archive_url": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip",
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
