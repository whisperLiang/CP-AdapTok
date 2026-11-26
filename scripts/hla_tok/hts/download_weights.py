"""Fetch the four published DCVC-UF checkpoints and verify OneDrive metadata."""

import argparse
import hashlib
import json
import shutil
import zipfile
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

import numpy as np

SHARE = ("https://1drv.ms/f/c/2866592d5c55df8c/"
         "IgAalzb_985lQ79GkXyW2P5OASPpZHHcrcGWEVQxO-mQCVg?e=qyvMN6")
WEIGHTS = ("image", "ld", "hts", "htl")
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "Chrome/130.0.0.0 Safari/537.36")
DEFAULT_WEIGHTS = Path("checkpoints/hla_tok/hts/official")
RANGE_BYTES = 16 * 1024**2


def filename(kind: str) -> str:
    return f"cvpr2026_{kind if kind == 'image' else 'video_' + kind}.pth.tar"


def file_hashes(path: Path) -> dict:
    """Microsoft QuickXorHash uses a 160-bit ring and a shift of 11 per byte."""
    digest = hashlib.sha256()
    quick = np.zeros(160, dtype=np.uint8)
    length = 0
    with path.open("rb") as handle:
        while data := handle.read(160 * 32768):
            length += len(data)
            digest.update(data)
            padded = data + bytes(-len(data) % 160)
            blocks = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 160)
            quick ^= np.bitwise_xor.reduce(blocks, axis=0)
    value = 0
    for index, byte in enumerate(quick):
        shift = (11 * index) % 160
        value ^= int(byte) << shift
    value = (value & (2**160 - 1)) ^ (value >> 160) ^ (length << 96)
    return {"bytes": length, "sha256": digest.hexdigest(),
            "quickxor": b64encode(value.to_bytes(20, "little")).decode()}


def official_files() -> dict:
    import requests

    # Resolve the published share through Microsoft's anonymous browser flow.
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        landing = session.get(SHARE, timeout=30)
        landing.raise_for_status()
        redeem = parse_qs(urlsplit(landing.url).query)["redeem"][0]
        token = session.post("https://api-badgerp.svc.ms/v1.0/token", timeout=30,
                             headers={"AppId": "1141147648"},
                             json={"appId": "5cbed6ac-a083-4e14-b191-b4ba07653de2"})
        token.raise_for_status()
        api = f"https://my.microsoftpersonalcontent.com/_api/v2.0/shares/u!{redeem}/driveitem/children"
        listing = session.get(api, timeout=30,
                              headers={"Prefer": "autoredeem", "Authorization": "Badger " + token.json()["token"]})
        if not listing.ok:
            raise RuntimeError(f"Microsoft OneDrive listing returned HTTP {listing.status_code}")
    return {row["name"]: row for row in listing.json()["value"] if "file" in row}


def ranged_download(url: str, size: int, prefix_size: int, temporary: Path) -> list[Path]:
    import requests

    ranges = [(start, min(start + RANGE_BYTES, size) - 1) for start in range(prefix_size, size, RANGE_BYTES)]

    def fetch(interval: tuple[int, int]) -> Path:
        first, last = interval
        path = temporary / f"{first}.part"
        for _ in range(3):
            try:
                with requests.get(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={first}-{last}"},
                                  timeout=(30, 120), stream=True) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != (
                            f"bytes {first}-{last}/{size}"):
                        raise ValueError(f"HTTP {response.status_code} or inconsistent Content-Range")
                    with path.open("wb") as handle:
                        for chunk in response.iter_content(1024**2):
                            handle.write(chunk)
                if path.stat().st_size == last - first + 1:
                    print(f"received {first}-{last} of {size}", flush=True)
                    return path
            except (requests.RequestException, ValueError, OSError):
                continue
        raise RuntimeError(f"weight download failed for byte range {first}-{last}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(fetch, ranges))


def verify(path: Path, row: dict) -> dict:
    result = file_hashes(path)
    if result["bytes"] != row["size"] or result["quickxor"] != row["file"]["hashes"]["quickXorHash"]:
        raise ValueError(f"official size or QuickXorHash mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"checkpoint ZIP checksum failed: {path.name}")
    return result | {"onedrive_item_id": row["id"]}


def download(path: Path, row: dict) -> dict:
    if path.is_file():
        return verify(path, row)
    prefix = path.with_suffix(path.suffix + ".part")
    prefix_size = prefix.stat().st_size if prefix.is_file() else 0
    with TemporaryDirectory(dir=path.parent) as temp_dir:
        temp = Path(temp_dir)
        parts = ranged_download(row["@content.downloadUrl"], row["size"], prefix_size, temp)
        combined = temp / "combined"
        with combined.open("wb") as output:
            for part in ([prefix] if prefix_size else []) + parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 1024**2)
        identity = verify(combined, row)
        combined.replace(path)
    prefix.unlink(missing_ok=True)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    published = official_files()
    args.weights.mkdir(parents=True, exist_ok=True)
    identity = {}
    for kind in WEIGHTS:
        name = filename(kind)
        identity[kind] = {"filename": name, **download(args.weights / name, published[name])}
        print(name, identity[kind], flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"official_share": SHARE, "weights": identity}, indent=2) + "\n")


if __name__ == "__main__":
    main()
