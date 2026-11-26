"""Attribute a Base cap failure before selecting a checkpoint for strict replay."""

import argparse
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from hla_tok.uf_v2.runtime import BASE_STANDARD_CAP
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.baseline import distribution
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v3.endpoint import load_endpoint
from scripts.hla_tok.v2.protocol import setup


def measure(assets, checkpoint, reader, clips):
    codec = load_endpoint(assets, checkpoint)
    rows = []
    for clip in clips:
        codec.reset()
        for index in range(clip.length):
            intra, reset = schedule(index)
            rgb = source_frame(reader, clip, index)
            frame = codec.native.encode(rgb, index, codec.qp[0 if intra else 1], intra, reset)
            size = APP_BASE_OVERHEAD + len(frame.payload)
            rows.append({"video": clip.video, "index": index, "type": "I" if intra else "P",
                         "bytes": size, "native_y_z_bytes": len(frame.payload), "over_cap": size > BASE_STANDARD_CAP})
    return rows


def summarize(rows):
    tables = {}
    for kind in ("I", "P"):
        selected = [row for row in rows if row["type"] == kind]
        tables[kind] = distribution([row["bytes"] for row in selected], "bytes") | {
            "frames": len(selected), "over": sum(row["over_cap"] for row in selected)}
    maximum = max(rows, key=lambda row: row["bytes"])
    return {"by_type": tables, "worst": maximum, "maximum": maximum["bytes"], "over_cap": sum(
        row["over_cap"] for row in rows), "status": "PASS" if maximum["bytes"] <= BASE_STANDARD_CAP else "FAIL"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    assets, (reader, plans) = OfficialAssets.verify(args.weights), inputs()
    stats = {}
    for path in args.checkpoints:
        rows = measure(assets, path, reader, plans[args.split])
        stats[path.stem] = summarize(rows) | {"sha256": sha256_file(path)}
        write_csv(args.output / f"{path.stem}.csv", rows)
        print(path.stem, stats[path.stem]["status"], stats[path.stem]["maximum"], flush=True)
    write_json(args.output / "summary.json", {"candidates": stats, "read_counts": reader.counts,
        "split": args.split, "cap": BASE_STANDARD_CAP,
        "scope": "diagnostic full native payload size; oversized Base is rejected by the strict endpoint"})


if __name__ == "__main__":
    main()
