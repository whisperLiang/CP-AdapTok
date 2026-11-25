"""Calibrate a separate 4096-byte Base-Lite quantization family using fixed gains."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedRefinement
from hla_tok.uf.official import NativeDecoder, NativeEncoder, OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from hla_tok.uf_v2.native_weights import LiteProfile
from hla_tok.uf_v2.runtime import Unified
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.base_sweep import report
from scripts.hla_tok.uf_v2.evaluate import quality_replay, summarize_quality
from scripts.hla_tok.v2.protocol import setup

CANDIDATES = ((1.0, 1.0), (1.4, .85), (1.8, .70), (2.5, .55), (3.0, .45))
CAP = 4096


@torch.no_grad()
def run(assets, reader, clips, profile, metric):
    encoder, decoder = NativeEncoder(assets), NativeDecoder(assets)
    profile.install(encoder)
    profile.install(decoder)
    rows = []
    for clip in clips:
        encoder.reset()
        decoder.reset()
        for index in range(clip.length):
            rgb = source_frame(reader, clip, index)
            intra, reset = schedule(index)
            torch.cuda.synchronize()
            start = perf_counter()
            base = encoder.encode(rgb, index, 0, intra, reset)
            decoded = decoder.decode(base)
            elapsed = (perf_counter() - start) * 1000
            row = {"video": clip.video, "frame": index, "type": "I" if intra else "P",
                   "application_bytes": len(base.payload) + APP_BASE_OVERHEAD, "total_ms": elapsed}
            if metric is not None:
                metric.prepare(rgb)
                row |= metric.score(decoded)
            rows.append(row)
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    trials = []
    for i_step, p_gain in CANDIDATES:
        profile = LiteProfile(i_step, p_gain)
        rows = run(assets, reader, plans["calibration"], profile, None)
        record = {"intra_step": i_step, "inter_gain": p_gain, **report(rows, CAP)}
        trials.append(record)
        print(profile.identity(), record["hard_cap_pass"], flush=True)
        write_csv(args.output / f"trial-{len(trials)}.csv", rows)
        if record["hard_cap_pass"]:
            break
    final = trials[-1]
    extras = {}
    if final["hard_cap_pass"]:
        profile = LiteProfile(final["intra_step"], final["inter_gain"])
        metric = NativeQuality()
        rows = run(assets, reader, plans["validation"], profile, metric)
        validation = report(rows, CAP) | {"complete_base_quality": summarize(rows)}
        extras["validation"] = validation
        write_csv(args.output / "validation_base.csv", rows)
        if validation["hard_cap_pass"]:
            codec = Unified(assets, UnifiedRefinement().cuda().eval(), (0, 0), base_cap=CAP,
                            family_id="lite", lite=profile)
            budgets = [0, .5, 1, 1.395, 2, 3, 3.594, 4, 5]
            quality = quality_replay(codec, assets, reader, plans["validation"], budgets, 256)
            write_csv(args.output / "quality.csv", quality)
            extras["quality"] = summarize_quality(quality, budgets)
            extras["prefix_family"] = codec.tag.hex()
    status = "PASS" if extras.get("validation", {}).get("hard_cap_pass") else "FAIL"
    write_json(args.output / "summary.json", {
        "trials": trials, "selected_on_calibration": final["hard_cap_pass"], "status": status,
        "read_counts": reader.counts, "physical_encodes_per_frame_per_candidate": 1, **extras,
    })


if __name__ == "__main__":
    main()
