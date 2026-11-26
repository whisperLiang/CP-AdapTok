"""Apply the frozen R0 continuation rules to physical paired streams."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.uf_v3.train import CONFIG
from scripts.hla_tok.v2.protocol import csv_at, json_at


def evidence(selected, reference):
    rows = {}
    for row in csv_at(reference):
        if row["scenario"] == "C3" and row["budget"] in ("5", "10", "20", "40"):
            rows[row["video"], row["index"], int(row["budget"])] = row
    pairs = []
    for item in csv_at(selected):
        key = item["video"], item["index"], int(float(item["budget_mbps"]))
        if key not in rows or item["status"] != "PASS":
            continue
        matched = rows[key]
        pairs.append({"video": key[0], "index": key[1], "budget": key[2],
                      **{metric: float(item[metric]) - float(matched[metric])
                         for metric in ("psnr", "lpips", "dists")}})
    return pairs


def compare(candidate, reference, config):
    by_cut = {int(row["budget_mbps"]): row for row in candidate["quality"] if row["status"] == "PASS"}
    cuts = (5, 10, 20, 40)
    if set(cuts) - set(by_cut):
        return {"status": "FAIL", "reason": "incomplete physical pilot", "seed2": "NOT_RUN"}
    quality = {b: {metric: by_cut[b][metric] - reference[f"C3/{b}"][metric]
                   for metric in ("global_psnr", "lpips", "dists")} for b in cuts}
    low = all(by_cut[int(b)]["global_psnr"] >= limit["min_psnr"]
              and by_cut[int(b)]["lpips"] <= limit["max_lpips"] for b, limit in config["pilot_low_rate"].items())
    rule = config["pilot_high_rate_signal"]
    high = any((quality[b]["global_psnr"] >= rule["gain_psnr_db"] and quality[b]["lpips"] <= rule["max_lpips_increase"])
               or (quality[b]["lpips"] <= -rule["gain_lpips"]
                   and quality[b]["global_psnr"] >= -rule["max_psnr_drop_db"]) for b in (20, 40))
    runtime = candidate.get("timing")
    realtime = bool(runtime) and all(t["pooled"]["fps"] >= 30 and t["pooled"]["total_ms_p95"] <= 33.33
                                    for t in runtime.values())
    passed = low and high and realtime
    return {"status": "PASS" if passed else "FAIL", "low_protection": low, "high_improvement": high,
            "realtime": realtime, "delta": quality, "seed2": "NOT_RUN",
            "reason": "three frozen pilot gates pass" if passed else "thresholds retained after physical replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json_at(args.candidate / "summary.json")
    reference = json_at(args.reference / "capacity.json")["results"]["2"]["pooled"]
    result = compare(candidate, reference, json_at(CONFIG))
    pairs = evidence(args.candidate / "quality.csv", args.reference / "seed2/frames.csv")
    if pairs:
        write_csv(args.output.parent / "pilot_paired.csv", pairs)
    write_json(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
