"""Verified resumable range download for networks that interrupt the large Hub object."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from .model_adapter import DEFAULT_REVISION, REPO_ID
from .video_reader import file_sha256


def download_part(root, url, start, end):
    target = root / f"part_{start:012d}"
    if target.exists() and target.stat().st_size == end - start + 1:
        return target
    for attempt in range(8):
        try:
            request = urllib.request.Request(url + f"?part={start}", headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
                    raise ValueError("Hub did not return the requested exact byte range")
                content = response.read()
            if len(content) != end - start + 1:
                raise ValueError("incomplete range response")
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(target)
            print(f"verified length: bytes {start}-{end}", flush=True)
            return target
        except (OSError, ValueError):
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    info_url = f"https://huggingface.co/api/models/{REPO_ID}/revision/{args.revision}?blobs=true"
    info = json.load(urllib.request.urlopen(info_url, timeout=30))
    revision = info["sha"]
    entry = next(x for x in info["siblings"] if x["rfilename"] == "model.safetensors")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/"
    config = urllib.request.urlopen(base + "config.json", timeout=30).read()
    (root / "config.json").write_bytes(config)
    final = root / "model.safetensors"
    if final.exists() and file_sha256(final) == entry["lfs"]["sha256"]:
        print("existing official checkpoint verified", flush=True)
        return
    parts = root / ".download_parts"
    parts.mkdir(exist_ok=True)
    size, chunk = entry["size"], 4 * 2**20
    ranges = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        paths = list(executor.map(lambda r: download_part(parts, base + "model.safetensors", *r), ranges))
    temporary = root / "model.safetensors.assembled"
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for path in paths:
            data = path.read_bytes()
            digest.update(data)
            handle.write(data)
    if digest.hexdigest() != entry["lfs"]["sha256"]:
        raise ValueError("download SHA256 does not match the official Hub object")
    temporary.replace(final)
    for path in paths:
        path.unlink()
    parts.rmdir()
    print(json.dumps({"revision": revision, "sha256": digest.hexdigest(), "bytes": size}), flush=True)


if __name__ == "__main__":
    main()
