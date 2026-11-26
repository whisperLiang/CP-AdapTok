"""Save a traced native source and physically decoded sample at each hard budget."""

import argparse
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.diagnostics import save_image
from scripts.hla_tok.uf_v2.evaluate import encode, receive
from scripts.hla_tok.uf_v3.endpoint import load_endpoint
from scripts.hla_tok.v2.protocol import setup
from third_party.DCVC.src.utils.transforms import rgb2ycbcr


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    clip = plans["validation"][0]
    codec = load_endpoint(OfficialAssets.verify(args.weights), args.checkpoint,
                          static=StaticAsset.load(args.static) if args.static else None, raw=args.raw)
    names = ("Source", "Base", *(f"budget_{b}" for b in BUDGETS))
    endpoints = {name: codec.new_decoder() for name in names[1:]}
    target = 5
    previous, temporal = {}, []
    for index in range(target + 1):
        rgb = source_frame(reader, clip, index)
        encoded = encode(codec, rgb, index)
        family = encoded.family
        images = {"Source": rgb}
        for (name, endpoint), cap in zip(endpoints.items(), (family.base_end, *BUDGETS.values()), strict=True):
            result, stats = receive(codec, endpoint, family, cap)
            if result is None or not codec.matches(encoded, endpoint, result[0]) or stats["bytes"] > cap:
                raise RuntimeError("sample could not be received independently at its declared cap")
            images[name] = result[1]
        source_luma = rgb2ycbcr(rgb)[:, :1]
        residual = {name: rgb2ycbcr(images[name])[:, :1] - source_luma for name in endpoints}
        if previous:
            temporal.append({"index": index, **{name: (value - previous[name]).square().mean().item()
                                                 for name, value in residual.items()}})
        previous = residual
    rows = [{"scenario": name, **save_image(args.output / f"{name}.png", images[name])} for name in names]
    stream = args.output / "maximum.huf"
    stream.write_bytes(family.raw)
    write_json(args.output / "manifest.json", {
        "video": clip.video, "index": target, "original_height_width": [1080, 1920], "images": rows,
        "crop_yxhw": [400, 820, 224, 224], "source_content_sha256": clip.manifest(reader)["frames"][target]["sha256"],
        "checkpoint_sha256": sha256_file(args.checkpoint), "bitstream_sha256": sha256_file(stream),
        "bitstream_bytes": len(family.raw), "prefix_bytes": {b: len(family.prefix(cap)[0]) for b, cap in BUDGETS.items()},
        "base_bytes": family.base_end, "model_tag": codec.tag.hex(), "read_counts": reader.counts,
        "native_temporal_luma_residual_change_mse": temporal,
        "temporal_scope": "5 adjacent pairs on the sample clip, without motion compensation",
    })


if __name__ == "__main__":
    main()
