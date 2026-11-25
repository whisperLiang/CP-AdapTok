"""Time the CPU integer add in the native decoder on archived, one-record P appends."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hla_tok.evaluation.io import write_json
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.latent import VIDEO_CHANNELS, UnifiedRefinement
from hla_tok.uf.official import sha256_file
from hla_tok.uf_v2.records import BatchRecords
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.uf.baseline import distribution


def profile_sample(bank, stream, ends, take_update_ns):
    height, width = NATIVE_SHAPE
    shape = (VIDEO_CHANNELS, (height + 15) // 16, (width + 15) // 16)
    full = bank.canvas(shape, False)
    offset, records = full.decode(bytearray(stream[ends[0]:]))
    if offset != ends[-1] - ends[0] or records != len(ends) - 1:
        raise ValueError("full frame record boundaries differ from the archived manifest")
    packets = [bytearray(stream[start:stop]) for start, stop in zip(ends, ends[1:])]
    elapsed = []
    take_update_ns()
    for rep in range(6):
        serial = bank.canvas(shape, False)
        for packet in packets:
            if serial.decode(packet) != (len(packet), 1):
                raise ValueError("the supplied enhancement range is not a single record")
            ns = take_update_ns()
            if rep > 0:
                elapsed.append(ns)
        if not np.array_equal(serial.symbols, full.symbols):
            raise ValueError("profiled increments did not reconstruct the same integer latent")
    return {"records_per_repeat": records, "passes": 5, "warmup_passes": 1,
            "integer_update": distribution(elapsed, "ns"), "integer_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--entropy-table", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import hla_uf_v2_records

    model = UnifiedRefinement().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["quantizer"])
    table = StaticAsset.load(args.entropy_table).tables if args.entropy_table else None
    bank = BatchRecords(tuple(model.scales().tolist()), Segmentation(8, 8), table)
    manifest = json.loads((args.samples / "manifest.json").read_text())
    rows = []
    for sample in manifest["samples"]:
        if sample["index"] != 1:
            continue
        source = args.samples / sample["path"]
        if sha256_file(source) != sample["sha256"]:
            raise ValueError("native sample checksum mismatch")
        rows.append({"sample": source.name, "sha256": sample["sha256"],
                     **profile_sample(bank, source.read_bytes(), sample["boundaries"],
                                      getattr(hla_uf_v2_records, "take_canvas_update_ns"))})
    if not rows:
        raise ValueError("requires archived continuous P frames")
    write_json(args.output, {"rows": rows, "scope": "native integer add only, steady_clock instrumentation in C++",
                             "checkpoint_sha256": sha256_file(args.checkpoint)})


if __name__ == "__main__":
    main()
