"""Attribute observed Base costs with a physical standalone encode of the same z symbols."""

import argparse
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.official import OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from hla_tok.uf_v2.native_weights import LiteProfile
from hla_tok.uf_v2.runtime import Analysis
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.baseline import distribution
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v3.lite import AttenuatedIntra
from scripts.hla_tok.v2.protocol import setup


def encode_z(coder, frame, symbols):
    ordered = symbols.permute(0, 2, 3, 1).contiguous().to(dtype=torch.int8, device="cpu").numpy().ravel()
    channels = symbols.shape[1]
    coder.reset()
    coder.set_entropy_coder_parallel(frame.info.parallel)
    coder.encode_z(ordered, frame.info.qp * channels, channels)
    coder.flush()
    return len(coder.get_encoded_stream())


@torch.no_grad()
def run(assets, reader, clips, profile):
    from MLCodec_extensions_cpp import RansEncoder

    native = Analysis(assets)
    profile.install(native)
    coders = {True: RansEncoder(), False: RansEncoder()}
    for intra, model in ((True, native.intra), (False, native._video())):
        coders[intra].set_cdf(*model.bit_estimator_z.get_cdf_info(), 0)
    rows = []
    for clip in clips:
        native.reset()
        for index in range(clip.length):
            intra, reset = schedule(index)
            base = native.encode_base(source_frame(reader, clip, index), index, (0, 0), intra, reset, 16384)
            with torch.cuda.stream(native.stream):
                snapshot = native.proxy(intra).uf_encode_snapshot()
            native.stream.synchronize()
            y = snapshot[4]
            z = snapshot[3]
            rows.append({"video": clip.video, "index": index, "type": "I" if intra else "P",
                         "base_bytes": len(base.payload) + APP_BASE_OVERHEAD, "native_y_z_bytes": len(base.payload),
                         "standalone_z_bytes": encode_z(coders[intra], base, z),
                         "official_parallel": base.info.parallel, "z_symbols": z.numel(),
                         "nonzero_z": int(torch.count_nonzero(z)), "nonzero_y_symbols_snapshot": int(
                             torch.count_nonzero(y))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    setup()
    assets = OfficialAssets.verify(options.weights)
    reader, clips = inputs()
    outcome = {}
    for name, profile in (("L0", LiteProfile(1, 1)), ("L1", AttenuatedIntra())):
        rows = run(assets, reader, clips["validation"], profile)
        write_csv(options.output / f"{name}_hyperprior.csv", rows)
        outcome[name] = {kind: {lane: distribution([row[lane] for row in rows if row["type"] == kind], lane)
                                for lane in ("native_y_z_bytes", "standalone_z_bytes")} for kind in ("I", "P")}
    write_json(options.output / "hyperprior.json", {"profiles": outcome, "read_counts": reader.counts,
        "scope": "same native z symbols and CDF independently terminated; y and z share a production rANS state",
        "additive_production_breakdown": "NOT_AVAILABLE"})


if __name__ == "__main__":
    main()
