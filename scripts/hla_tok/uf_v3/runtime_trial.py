"""Compare a temporary CPU packing workspace on a clean, matched 40 Mbps replay."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import csv_at, json_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = {}
    quality = {}
    for mode in ("original", "reuse"):
        directory = args.folder / mode
        data = json_at(directory / "summary.json")
        row = data["timing"]["40"]["pooled"]
        report[mode] = {"fps": row["fps"], "p95_ms": row["total_ms_p95"], "p99_ms": row["total_ms_p99"],
                        "pack_ms": row["group_rank_pack_ms_mean"], "other_gpu": data["other_gpu_compute_before_after"]}
        quality[mode] = [tuple(row[field] for field in ("video", "index", "bytes", "records", "mse", "lpips", "dists"))
                         for row in csv_at(directory / "quality.csv")]
    equivalent = quality["original"] == quality["reuse"]
    improvement = report["original"]["p95_ms"] - report["reuse"]["p95_ms"]
    write_json(args.folder / "summary.json", {
        "same_physical_bytes_counts_and_rgb_quality": equivalent, "measurements": report,
        "p95_change_ms": -improvement, "minimal_investigation_threshold_ms": 1,
        "threshold_signal": equivalent and improvement >= 1 and report["reuse"]["p95_ms"] <= 33.33,
        "scope": "three paired native Calibration repetitions at 40 Mbps, local CPU pack candidate",
        "deployment": "Calibration only; the deployment choice is recorded in selection.json",
    })


if __name__ == "__main__":
    main()
