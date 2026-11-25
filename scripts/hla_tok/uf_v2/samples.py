"""Save identical-source native 5 Mbps reconstructions with physical-byte provenance."""
import argparse
from copy import copy
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.official import OfficialAssets, sha256_file
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.diagnostics import save_image
from scripts.hla_tok.uf_v2.evaluate import decoder_for, encode, load_codec, receive
from scripts.hla_tok.v2.protocol import setup


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--entropy-table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.precision, args.fixed, args.sensitivity, args.raw = "bf16", False, None, False
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    clip = plans["validation"][0]
    rgb = [source_frame(reader, clip, index) for index in range(6)]
    rows = [save_image(args.output / "source.png", rgb[-1])]
    for method in ("anchor-fast", "d0-fast", "d1"):
        options = copy(args)
        options.method = method
        if method != "d1":
            options.entropy_table = None
            options.checkpoint = args.anchor if method == "anchor-fast" else None
        codec = load_codec(options, assets)
        decoder = decoder_for(codec, assets)
        for index, image in enumerate(rgb):
            family = encode(codec, image, index).family
            rendered, stats = receive(codec, decoder, family, BUDGETS[5])
            if rendered is None:
                raise ValueError("missing sample reconstruction")
        raw, _ = family.prefix(BUDGETS[5])
        path = args.output / f"{method}.huf"
        path.write_bytes(raw)
        rows.append(save_image(args.output / f"{method}.png", rendered[1]) | {
            "application_bytes": stats["bytes"], "bitstream_sha256": sha256_file(path),
            "bitstream": path.name, "full_family_bytes": len(family.raw),
            "checkpoint_sha256": sha256_file(options.checkpoint) if options.checkpoint else None,
        })
    write_json(args.output / "manifest.json", {
        "video": clip.video, "index": 5, "cap_bytes": BUDGETS[5], "crop_xyxy": [730, 390, 1114, 606],
        "rgb_conversion": "native RGB quantized to 8-bit PNG, shared unmodified display range",
        "samples": rows, "read_counts": reader.counts,
    })


if __name__ == "__main__":
    main()
