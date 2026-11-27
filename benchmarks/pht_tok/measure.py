"""Final runtime replay and FIXED control on the same UVG RGB display target."""

import argparse
import json
from pathlib import Path

import torch

from baselines.dcvc_uf.native import NativeNetworks
from benchmarks.pht_tok.closed import official_profile, profile_report, run_closed
from benchmarks.pht_tok.study import (
    ROOT,
    SKIP,
    Quality,
    aggregate_curves,
    fixed_point,
    uvg_source,
)
from common.artifacts import write_json
from pht_tok.bitstream import Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from third_party.DCVC.src.utils.common import set_torch_env


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", choices=("profile", "fixed-uvg"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--uvg", type=Path, required=True)
    p.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    args = p.parse_args()
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        model = model_identifier(args.weights, SKIP)
        tx = PHTEncoder(NativeNetworks(args.weights, "hts", SKIP), model)
        rx = PHTDecoder(NativeNetworks(args.weights, "hts", SKIP), model)
        stock = NativeNetworks(args.weights, "hts", SKIP)
        source = uvg_source(args.uvg, 129)
        if args.study == "fixed-uvg":
            rows = fixed_point(source, 32, stock, tx, rx, Quality(), sample_dir=None, estimate_rate=True)
            write_json(args.work / "fixed-uvg.json", {"curves": aggregate_curves(rows), "units": rows})
            return
        stock_profiles = [official_profile(source, stock, 32) for _ in range(3)]
        candidate = {}
        for pattern in ("base", "mid", "full", "incremental"):
            repetitions = []
            for _ in range(3):
                outcome = run_closed(source, 32, tx, rx, Mode.MASK, 2,
                                     "full" if pattern == "incremental" else pattern,
                                     render_incrementally=pattern == "incremental")
                repetitions.append(profile_report(outcome["per_unit"], source, outcome["process_cpu_seconds"]))
            candidate[pattern] = repetitions
            print("optimized", pattern, flush=True)
        first = run_closed(source, 32, tx, rx, Mode.MASK, 1, "base")
        first_profile = profile_report(first["per_unit"], source, first["process_cpu_seconds"])
        before = json.loads((args.work / "profiling-before.json").read_text())
        write_json(args.work / "profiling.json", {
            "official": stock_profiles[0], "official_repetitions": stock_profiles,
            "mask_kref2": candidate, "first_legal_prefix_kref1": first_profile, "before": before,
            "sampling": "1920x1080 UVG, budget 2 reference, 4 warm-up P chunks, 3 repetitions, batch 1",
            "capture_wait_at_30_fps_ms": 7000 / 30,
        })


if __name__ == "__main__":
    main()
