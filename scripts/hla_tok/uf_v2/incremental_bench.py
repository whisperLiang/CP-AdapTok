"""Measure single-record ingest and cumulative rendering on four native access units."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.official import OfficialAssets
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.baseline import distribution
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import encode, load_codec, receive
from scripts.hla_tok.v2.protocol import setup


def explicit_replay(codec, decoder, family, index):
    receiver = codec.receiver(decoder)
    clock = perf_counter()
    receiver.feed(family.raw[:family.base_end])
    if receiver.decode_available() != 1:
        raise ValueError("expected exactly one complete native Base")
    base_ms = (perf_counter() - clock) * 1e3
    targets = {count: budget for budget, cap in BUDGETS.items() for _, count in (family.prefix(cap),)}
    records, renders = [], []
    last_image = receiver.render()
    for n, (start, stop) in enumerate(zip(family.ends, family.ends[1:]), 1):
        payload = family.raw[start:stop]
        tick = perf_counter()
        receiver.feed(payload)
        received = receiver.decode_available()
        ingest_ms = (perf_counter() - tick) * 1e3
        if received != 1 or receiver.decode_available() != 0:
            raise ValueError("incremental enhancement was not newly handled exactly once")
        records.append({"index": index, "record": n, "bytes": len(payload), "ingest_ms": ingest_ms})
        if n in targets:
            tick = perf_counter()
            last_image = receiver.render()
            torch.cuda.synchronize()
            renders.append({"index": index, "budget": targets[n], "records": n,
                            "display_ms": (perf_counter() - tick) * 1e3})
    receiver.close()
    return last_image, base_ms, records, renders


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--entropy-table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.method, options.precision, options.fixed, options.raw = "d1", "bf16", False, False
    options.sensitivity = None
    setup()
    reader, plans = inputs()
    codec = load_codec(options, OfficialAssets.verify(options.weights))
    full, incremental = codec.new_decoder(), codec.new_decoder()
    records, renders, frames = [], [], []
    target = plans["validation"][0]
    images = [source_frame(reader, target, index) for index in range(4)]
    for index, rgb in enumerate(images):
        encoded = encode(codec, rgb, index)
        for endpoint in (full, incremental):
            warm, _ = receive(codec, endpoint, encoded.family, BUDGETS[40])
            if warm is None:
                raise ValueError("incremental timing warmup requires a complete prefix")
    codec.reset()
    full.reset()
    incremental.reset()
    for index, rgb in enumerate(images):
        encoded = encode(codec, rgb, index)
        complete, stats = receive(codec, full, encoded.family, BUDGETS[40])
        if complete is None:
            raise ValueError("native full prefix failed")
        image, base_ms, new, redraw = explicit_replay(codec, incremental, encoded.family, index)
        if not torch.equal(image, complete[1]):
            raise ValueError("incremental render differs from one-shot final render")
        records.extend(new)
        renders.extend(redraw)
        frames.append({"index": index, "base_available_ms": base_ms,
                       "total_one_record_ingest_ms": sum(x["ingest_ms"] for x in new),
                       "total_multi_render_ms": sum(x["display_ms"] for x in redraw),
                       "one_render_ms": stats["render_ms"], "incremental_exact": True})
    write_csv(options.output / "append.csv", records)
    write_csv(options.output / "render.csv", renders)
    write_json(options.output / "summary.json", {"frames": frames, "read_counts": reader.counts,
               "one_record_ingest": distribution((row["ingest_ms"] for row in records), "ms"),
               "records": len(records), "model_tag": codec.tag.hex(), "warmup": len(images),
               "note": "record ingest combines wire parser, entropy decoding and integer canvas update"})


if __name__ == "__main__":
    main()
