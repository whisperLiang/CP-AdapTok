"""Verify Original AdapTok bytes and compare already measured curves at actual rates."""

import argparse
from collections import defaultdict
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.continuous.quality import summarize
from scripts.hla_tok.uf_v2.history import original
from scripts.hla_tok.uf_v2.rate_analysis import at_rate
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

METRICS = ("global_psnr", "lpips", "dists")


def original_scope():
    archived = ROOT / "results/hla_tok/compact"
    scope = json_at(archived / "original_archive_identity.json")
    identity = original(ROOT, scope["cohort_sha256"])
    quality = defaultdict(list)
    keys = set()
    for item in csv_at(archived / "original_per_frame.csv"):
        budget = int(item["budget_per_group"])
        key = (budget, item["video"], int(item["frame"]))
        if key in keys:
            raise ValueError("Original duplicate frame")
        keys.add(key)
        quality[budget].append({metric: float(item[metric]) for metric in ("mse", "psnr", "lpips", "dists")})
    expected = {(budget, video, frame) for budget in scope["budgets_per_group"]
                for video, count in scope["cohort"].items() for frame in range(count)}
    if keys != expected:
        raise ValueError("Original and UF frame identities differ")
    ledger = []
    for row in csv_at(archived / "original_rd.csv"):
        budget = int(row["budget_per_group"])
        observed = summarize(quality[budget])
        if any(abs(observed[metric] - float(row[metric])) > 1e-9 for metric in METRICS):
            raise ValueError("Original metric aggregation differs")
        ledger.append({"budget_per_group": budget, "quality": observed, "bytes": int(row["total_bytes"])})
    return {"identity": identity, "frame_keys_verified": len(keys), "ledger": ledger,
            "color": "RGB float32 [0,1], native 1080p before scoring; source conversion BGR→RGB",
            "spatial": "128×128 stride 128 tiles; matched coordinates stitched before scoring",
            "lpips": "AlexNet v0.1, RGB [0,1] mapped to [-1,1]; frame-macro average",
            "dists": "distributed weights; native RGB [0,1]; frame-macro average",
            "psnr": "global MSE over equal-sized frames",
            "temporal": json_at(archived / "original_temporal_scope.json"),
            "wire": "actual Fixed13 + 24 B F13W per 16-frame window; average over 174 valid frames",
            "hard_budget_equivalence": "NOT_COMPARABLE"}


def compare(rates, curves):
    rows = []
    for point in rates:
        for name in ("Official UF-LD", "Anchor-Fast"):
            rate = point["actual_mbps"]
            values = {metric: at_rate(curves[name], metric, rate) for metric in METRICS}
            rows.append({"capacity_budget": point["budget"], "actual_mbps": rate, "reference": name,
                         "status": "PASS" if values["lpips"] is not None else "OUT_OF_RANGE", **values})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results / "diagnostics"
    scope = original_scope()
    archive = json_at(ROOT / "results/hla_tok/uf_v2/evaluation/rate_analysis.json")["curves"]
    val = json_at(out / "validation/capacity.json")["results"]["2"]["pooled"]
    rows = compare([val[f"C3/{budget}"] | {"budget": budget} for budget in (2, 5, 10, 20, 40)], archive)
    write_csv(out / "same_actual_rate.csv", rows)
    write_json(out / "references.json", {"original": scope, "same_actual_rate": rows,
        "control_scope": "frozen actual native C5/C6 streams and log-rate interpolation within their observed ranges",
        "available_official_mbps": [min(row["actual_mbps"] for row in archive["Official UF-LD"]),
                                    max(row["actual_mbps"] for row in archive["Official UF-LD"])]})


if __name__ == "__main__":
    main()
