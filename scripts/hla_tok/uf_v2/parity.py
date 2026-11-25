"""Compare complete legacy and batched application streams on audited I and P frames."""

import argparse
import hashlib
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.uf.anchor import BUDGETS, AnchorModel, RefinementCodec
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedCodec, UnifiedRefinement
from hla_tok.uf.official import NativeDecoder, OfficialAssets
from hla_tok.uf_v2.runtime import AnchorFast, Unified
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.anchor_eval import AnchorEncoder
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf.unified_eval import UnifiedEncoder
from scripts.hla_tok.v2.protocol import setup


@torch.no_grad()
def parity(assets, legacy, fast, reader, clip, frames):
    decoder_a, decoder_b = NativeDecoder(assets), fast.new_decoder()
    digests, boundaries = [], 0
    for i in range(frames):
        rgb = source_frame(reader, clip, i)
        a = legacy.encode(rgb, i, *schedule(i))
        b = fast.encode(rgb, i, *schedule(i))
        if a.family != b.family or not torch.equal(a.reference, b.reference):
            raise ValueError(f"legacy/full-max parity failed at frame {i}")
        raw, _ = a.family.prefix(list(BUDGETS.values())[i % len(BUDGETS)])
        ra, rb = legacy.receiver(decoder_a), fast.receiver(decoder_b)
        ra.feed(raw)
        rb.feed(raw)
        if ra.decode_available() != rb.decode_available():
            raise ValueError("record count differs")
        ra.finish()
        rb.finish()
        if not torch.equal(ra.render(), rb.render()):
            raise ValueError(f"display output differs at frame {i}")
        if not fast.matches(b, decoder_b, rb):
            raise ValueError(f"reference differs at frame {i}")
        digests.append(hashlib.sha256(b.family.raw).hexdigest())
        boundaries += len(b.family.ends)
    return {"frames": frames, "boundaries": boundaries, "sha256": digests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=34)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    clip = plans["validation"][1]
    assets = OfficialAssets.verify(args.weights)
    anchor = AnchorModel().cuda().eval()
    anchor.load_state_dict(torch.load(args.anchor, map_location="cpu", weights_only=True)["model"])
    unified = UnifiedRefinement().cuda().eval()
    qp = (0, 8)
    pairs = {"anchor": (AnchorEncoder(assets, RefinementCodec(anchor, "bf16"), qp),
                        AnchorFast(assets, RefinementCodec(anchor, "bf16"), qp)),
             "d0": (UnifiedEncoder(assets, UnifiedCodec(unified), qp), Unified(assets, unified, qp))}
    rows = {method: parity(assets, *pair, reader, clip, args.frames) for method, pair in pairs.items()}
    write_json(args.output, {"status": "PASS", "rows": rows, "reads": reader.counts,
                             "clip": clip.video, "anchor": str(args.anchor)})


if __name__ == "__main__":
    main()
