"""Lawfully fetch complete REDS official Train Sharp sequences with bounded HTTP range caching."""

import argparse
import json
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

from cp_adaptok.large_scale_data import audit_source, load_manifests, write_manifests
from cp_adaptok.runtime import resolve, write_json
from scripts.prepare_optimization_data import verified_frame
from scripts.remote_archive import RangeReader

SOURCE = "https://seungjunnah.github.io/Datasets/reds.html"
URL = "https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip?download=true&cpat=20261116"


class BoundedRangeReader(RangeReader):
    def __init__(self, url):
        super().__init__(url, block_size=8 * 2**20)
        self.blocks = OrderedDict()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reds-http")
        self.pending = {}

    def fetch_block(self, index):
        import urllib.request

        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        return self._fetch(request, end - start + 1)

    def _block(self, index):
        if index in self.blocks:
            self.blocks.move_to_end(index)
            return self.blocks[index]
        while len(self.blocks) >= 8:
            self.blocks.popitem(last=False)
        # Only this reader thread mutates caches and counters. Workers return immutable bytes.
        if index not in self.pending:
            self.pending[index] = self.pool.submit(self.fetch_block, index)
        for following in range(index + 1, min(index + 4, (self.size + self.block_size - 1) // self.block_size)):
            if following not in self.blocks and following not in self.pending:
                self.pending[following] = self.pool.submit(self.fetch_block, following)
        data = self.pending.pop(index).result()
        self.downloaded += len(data)
        self.blocks[index] = data
        return data


def prepare(count, output):
    if not 1 <= count <= 240:
        raise ValueError("REDS Train Sharp contains 240 independent sequences")
    if shutil.disk_usage(output.parent).free < 25 * 2**30:
        raise RuntimeError("need at least 25 GiB free before bounded REDS preparation")
    reader = BoundedRangeReader(URL)
    records = []
    with ZipFile(reader) as archive:
        groups = {}
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if len(parts) == 4 and parts[:2] == ("train", "train_sharp") and parts[-1].endswith(".png"):
                groups.setdefault(parts[2], []).append(info)
        # Physical archive order reduces duplicate downloads; sequence identity is unchanged.
        names = sorted(groups, key=lambda name: min(x.header_offset for x in groups[name]))[:count]
        for name in names:
            # ZIP stores frames out of lexical order. Read by physical offset to avoid
            # evicting and redownloading the same ranges; filenames retain temporal order.
            entries = sorted(groups[name], key=lambda x: x.header_offset)
            if len(entries) != 100:
                raise ValueError(f"incomplete official REDS sequence {name}")
            destination = output / "train" / name
            destination.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                verified_frame(archive, entry.filename, destination / Path(entry.filename).name)
            row = audit_source(destination, f"REDS:train:{name}", SOURCE, "train", fps=24, fps_nominal=False)
            # These image sequences use the documented standard 24 fps sampling, not container timestamps.
            row.update(fps_note="official standard REDS Sharp 24 fps sampling; no per-frame timestamps",
                       fps_nominal=True)
            if row["resolution"] != [1280, 720]:
                raise ValueError(f"REDS source is not native 720p: {row['resolution']}")
            records.append(row)
            write_json(output / "preparation.json", {"records": records, "downloaded_bytes": reader.downloaded,
                                                      "license": "CC BY 4.0", "url": URL})
            print(f"REDS train {name}: 100 native 720p frames; {len(records)}/{count}; "
                  f"downloaded {reader.downloaded / 2**30:.2f} GiB", flush=True)
    splits, _ = load_manifests()
    existing = {row["content_id"] for row in splits["train"]}
    splits["train"].extend(row for row in records if row["content_id"] not in existing)
    previous = json.loads(resolve("data/manifests/cpat_train.json").read_text())["provenance"]
    provenance = dict(previous, reds={"source": SOURCE, "archive_url": URL, "license": "CC BY 4.0",
                                    "complete_sequences": len(records), "zip_crc_verified": True,
                                    "selection": "first complete sequences in archive physical order",
                                    "downloaded_bytes": reader.downloaded})
    return write_manifests(splits, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=60)
    parser.add_argument("--output", default="data/large_scale/reds")
    args = parser.parse_args()
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args.sequences, output), indent=2))


if __name__ == "__main__":
    main()
