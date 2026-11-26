"""Hash native full prefix bytes and every legal endpoint for backend parity."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import OfficialAssets
from hla_tok.uf_v2.static_entropy import StaticAsset
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import encode
from scripts.hla_tok.uf_v3.endpoint import load_endpoint
from scripts.hla_tok.v2.protocol import setup


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    setup()
    codec = load_endpoint(OfficialAssets.verify(options.weights), options.checkpoint,
                          static=StaticAsset.load(options.static))
    reader, plans = inputs()
    ledger = []
    for clip in plans["validation"]:
        codec.reset()
        for index in range(clip.length):
            family = encode(codec, source_frame(reader, clip, index), index).family
            ends = np.asarray(family.ends, dtype="<i4")
            ledger.append({"video": clip.video, "index": index, "sha256": hashlib.sha256(family.raw).hexdigest(),
                           "ends_sha256": hashlib.sha256(ends.tobytes()).hexdigest(), "bytes": len(family.raw),
                           "boundaries": len(ends), "base_bytes": family.base_end})
    write_json(options.output, {"frames": ledger, "model_tag": codec.tag.hex(), "read_counts": reader.counts})


if __name__ == "__main__":
    main()
