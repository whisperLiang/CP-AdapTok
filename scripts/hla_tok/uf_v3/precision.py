"""Probe the effect of a finer existing quantizer under the same Base and frozen E1 CDF."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import LATENT_UNIT, normalized_error, render_latent
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import MAX_FRAME_BYTES
from hla_tok.uf_v2.records import quantized_symbols
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import allowance
from scripts.hla_tok.uf_v3.capacity import DIAGNOSTIC_TAG, native_format
from scripts.hla_tok.uf_v3.endpoint import load_v2, parse_base, standard_decoder
from scripts.hla_tok.v2.protocol import setup

MODES = {"existing": 1.0, "half_step": 0.5}
CUTS = {"20": allowance(20), "40": allowance(40), "all": MAX_FRAME_BYTES}


def score_frame(codec, receiver, metric, clip, index, rgb):
    intra, reset = schedule(index)
    encoded = codec.native.encode_base(rgb, index, codec.qp, intra, reset, codec.base_cap)
    original, reference, scale = codec.native.proxy(intra).ufv2_encode_snapshot()
    continuous = original.float() * scale.float()
    error = normalized_error(original, reference, scale)
    receiver_state = None
    metric.prepare(rgb)
    rows = []
    for name, adjustment in MODES.items():
        step = codec.steps[intra] * adjustment
        symbols = quantized_symbols(error, step)
        family = codec.bank.pack_symbols(encoded, DIAGNOSTIC_TAG, symbols, MAX_FRAME_BYTES)
        if receiver_state is None:
            receiver_state = receiver.shared_base(parse_base(family, DIAGNOSTIC_TAG))
            if not torch.equal(receiver.decoded_latent(), reference):
                raise ValueError("the fixed-history control did not restore Base exactly")
        canvas = codec.bank.canvas(tuple(symbols.shape[1:]), intra)
        position = family.base_end
        for limit, cap in CUTS.items():
            raw, records = family.prefix(cap)
            used, _ = canvas.decode(bytearray(raw[position:]))
            if records != canvas.records or position + used != len(raw):
                raise ValueError("physical refinement replay differs from the advertised prefix")
            position += used
            residual = canvas.delta(scale.float() * step * LATENT_UNIT)
            code = reference.float() + residual
            rgb_out = render_latent(receiver, encoded.info, native_format(code), receiver_state.context)
            rows.append({"video": clip.video, "index": index, "mode": name, "budget": limit, "bytes": len(raw),
                         "records": records, "latent_mse": float((code - continuous).square().mean()),
                         **metric.score(rgb_out)})
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    codec = load_v2(OfficialAssets.verify(args.weights), args.checkpoint, static=StaticAsset.load(args.static))
    receiver = standard_decoder(codec)
    reader, plans = inputs()
    metric = NativeQuality()
    rows = []
    for clip in plans["calibration"]:
        codec.reset()
        receiver.reset()
        for index in range(12):
            rows.extend(score_frame(codec, receiver, metric, clip, index, source_frame(reader, clip, index)))
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"], row["budget"]].append(row)
    pooled = {f"{name}/{budget}": summarize(items) | {"actual_mbps": sum(row["bytes"] for row in items) * 240 / (
        1e6 * len(items)), "latent_mse": sum(row["latent_mse"] for row in items) / len(items)}
              for (name, budget), items in groups.items()}
    write_csv(args.output / "precision.csv", rows)
    write_json(args.output / "precision.json", {"read_counts": reader.counts, "checkpoint": sha256_file(args.checkpoint),
        "fixed_base": True, "scope": "Calibration D0 display control; existing two levels and E1 CDF; no optimization",
        "pooled": pooled})


if __name__ == "__main__":
    main()
