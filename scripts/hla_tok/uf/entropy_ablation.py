"""Measure static and conditional CDF bytes for the same transmitted integers."""

import argparse
from pathlib import Path

import numpy as np
import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.anchor import STRIDE, AnchorModel, RefinementCodec
from hla_tok.uf.base import BaseFrame
from hla_tok.uf.entropy import MULTIPLIERS
from hla_tok.uf.official import sha256_file
from hla_tok.uf.wire import RECORD_OVERHEAD, FrameParser
from scripts.hla_tok.v2.protocol import json_at


def summarize_packet(raw: bytes, codec: RefinementCodec) -> dict:
    parser = FrameParser(codec.tag)
    parser.feed(raw)
    records = parser.available()
    base = next(records)
    if not isinstance(base, BaseFrame):
        raise ValueError("ablation requires a complete native base")
    bank = codec.records_codec
    layout = bank.layout((codec.model.channels, base.info.height // STRIDE, base.info.width // STRIDE))
    total = {"records": 0, "conditional_bytes": 0, "static_bytes": 0, "record_header_bytes": 0}
    for record in records:
        if isinstance(record, BaseFrame):
            raise ValueError("unexpected second native base")
        count = int(np.prod(layout.extent(record.group)))
        symbols = bank.coder.decode(record.payload, record.cdf, count)
        static_cdf = record.level * len(MULTIPLIERS) + 1
        alternative = bank.coder.encode(symbols, static_cdf)
        if not np.array_equal(symbols, bank.coder.decode(alternative, static_cdf, count)):
            raise ValueError("CDF ablation changed the transmitted symbols")
        total["records"] += 1
        total["conditional_bytes"] += len(record.payload)
        total["static_bytes"] += len(alternative)
        total["record_header_bytes"] += RECORD_OVERHEAD
    parser.finish()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = AnchorModel().eval()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True)["model"])
    codec = RefinementCodec(model, "bf16")
    manifest = json_at(args.samples / "manifest.json")
    if sha256_file(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("sample/checkpoint identity mismatch")
    rows = []
    for entry in manifest["samples"]:
        path = args.samples / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"sample hash mismatch: {path}")
        rows.append({"file": entry["file"]} | summarize_packet(path.read_bytes(), codec))
    sums = {key: sum(row[key] for row in rows) for key in ("conditional_bytes", "static_bytes", "record_header_bytes")}
    write_json(args.output, {"samples": rows, "totals": sums, "same_symbols": True,
                "payload_saving_fraction": 1 - sums["conditional_bytes"] / sums["static_bytes"]})


if __name__ == "__main__":
    main()
