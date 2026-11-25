"""Replay fixed single-pass Base-Standard and Base-Lite families."""
import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.official import OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from scripts.hla_tok.continuous.quality import NativeQuality
from scripts.hla_tok.uf.baseline import Point, distribution, evaluate
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import setup

CAPS = {0.5: 2083, 1: 4166, 1.395: 5812, 2: 8333, 3: 12500, 3.594: 14975, 4: 16666,
        5: 20833, 10: 41666, 20: 83333, 40: 166666}
FAMILIES = {"base-standard": (0, 8, 9212), "base-lite": (0, 0, 4096)}


def report(rows, cap: int):
    by_type = {}
    for kind in ("I", "P"):
        group = [row["application_bytes"] for row in rows if row["type"] == kind]
        by_type[kind] = distribution(group, "bytes") | {
            "frames": len(group), "over_cap": sum(size > cap for size in group)}
    reach = {budget: sum(row["application_bytes"] > size for row in rows) for budget, size in CAPS.items()}
    return {"base_cap": cap, "hard_cap_pass": not any(row["over_cap"] for row in by_type.values()),
            "by_type": by_type, "unreachable_base_frames": reach,
            "actual_application_mbps": sum(row["application_bytes"] for row in rows) * 240 / len(rows) / 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "calibration", "long"), default="validation")
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets, quality = OfficialAssets.verify(args.weights), NativeQuality()
    scores = {}
    for family, (qi, qp, cap) in FAMILIES.items():
        rows, score = evaluate(assets, reader, plans[args.split], Point(family, qi, qp, 32), quality)
        for row in rows:
            row["application_bytes"] = row["payload_bytes"] + APP_BASE_OVERHEAD
            row["application_header_bytes"] = APP_BASE_OVERHEAD
            row["family_cap_exceeded"] = row["application_bytes"] > cap
        scores[family] = report(rows, cap) | {"native_complete_base": score, "qp": [qi, qp]}
        write_csv(args.output / f"{family}.csv", rows)
    write_json(args.output / "summary.json", {"families": scores, "read_counts": reader.counts,
               "manifest_hashes": reader.hashes, "hyperprior": "joint with native y rANS payload",
               "quality_scope": "complete base access units, including those over the family limit"})


if __name__ == "__main__":
    main()
