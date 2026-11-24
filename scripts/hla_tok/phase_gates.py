"""Derive phase status from persisted observations; start no experiments."""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hla_tok.evaluation.io import write_csv, write_json

CASES = {(128, "fp32"), (128, "bf16"), (96, "fp32"), (96, "bf16")}
GROUP_BUDGETS = {96, 128, 256, 384, 480, 512}


def csv_at(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_summary(path):
    if not path.is_file():
        return None
    tests = ET.parse(path).findall(".//testcase")
    failures = sum(case.find("failure") is not None or case.find("error") is not None for case in tests)
    skipped = sum(case.find("skipped") is not None for case in tests)
    return {"passed": len(tests) - failures - skipped, "failed": failures, "skipped": skipped}


def original_matched(rows):
    indices = {}
    for r in rows:
        if r["method"] != "Original AdapTok" or int(r["frames"]) != 174:
            continue
        if not (r.get("lpips") and r.get("dists") and int(r["total_bytes"]) > 0):
            continue
        indices.setdefault((int(r["stride"]), r["precision"]), set()).add(int(r["tokens_per_tile_block"]))
    return all(indices.get(key) == GROUP_BUDGETS for key in CASES)


def hardware_floor(bench):
    return {r["precision"]: float(r["neural_wall_fps"]) for r in bench
            if r["budget"] == "8192" and int(r["source_frames"]) >= 64 and int(r["repeats"]) >= 2}


def measured_phase0(root, benchmark, cpu, cuda):
    baselines = csv_at(root / "baselines/metrics.csv")
    measure = root / "baselines/joint_replay"
    replay = csv_at(measure / "metrics.csv")
    fps = hardware_floor(csv_at(benchmark / "benchmark.csv"))
    assets = root / "audit/final/assets.json"
    checks = {
        "matched_original_full_cohort": original_matched(baselines),
        "matched_joint_step40000": len(replay) == 8 and all(
            int(r["frames"]) == 174 and r["measurement"] == "verified_archived_bytes_native_rgb_replay" for r in replay
        ),
        "official_source_and_weights": (root / "baselines/official_weights.json").is_file(),
        "official_precision_tile_matrix": len(csv_at(root / "baselines/tile_batch.csv")) == 6,
        "official_4frame_latency": len(csv_at(root / "baselines/cache/four_frame_cache_latency.csv")) == 8,
        "official_compile_diagnostic": (root / "baselines/compile/official_compile.json").is_file(),
        "native_h100_performance": set(fps) == {"fp32", "bf16"} and fps["bf16"] >= 15,
        "cpu_regression": cpu is not None and cpu["failed"] == 0 and cpu["passed"] > 0,
        "cuda_native_tests": cuda is not None and cuda == {"passed": 3, "failed": 0, "skipped": 0},
        "sealed_old_assets": assets.is_file() and json.loads(assets.read_bytes())["verified_unchanged"],
    }
    status = "PASS" if all(checks.values()) else "PARTIAL PASS"
    if fps.get("bf16", float("inf")) < 15:
        status = "FAIL"
    return {"status": status, "checks": checks, "K8192_neural_serial_fps": fps}


def record_phase1(results):
    quality = results / "phase1/fixed/fixed_quality_gate.json"
    if not quality.is_file():
        return {"status": "BLOCKED", "reason": "fixed-budget native Validation has not completed"}
    measured = json.loads(quality.read_bytes())
    return {"status": "PARTIAL PASS" if measured["quality_pass"] else "FAIL", "quality": measured,
            "reason": "Phase 1 engineering and codebook acceptance must also be reviewed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/hla_tok"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("results/hla_tok/phase0/t0_initial"))
    args = parser.parse_args()
    results = args.results
    identity = json.loads(Path("configs/hla_tok/validation_cohort.json").read_bytes())
    cpu = test_summary(results / "phase0/cpu_regression.xml")
    cuda = test_summary(results / "phase0/cuda_tests.xml")
    phase0 = measured_phase0(results, args.benchmark_dir, cpu, cuda)
    pixel_reads = {"train_pixel_reads": 0, "validation_pixel_reads": 0, "test_pixel_reads": 0}
    for path in results.rglob("pixel_reads.json"):
        for name, count in json.loads(path.read_bytes()).items():
            pixel_reads[name] += count
    extra = results / "audit/initial/validation_pixels_audit.json"
    if extra.exists():
        for name, count in json.loads(extra.read_bytes()).items():
            pixel_reads[name] += count
    if pixel_reads["test_pixel_reads"]:
        phase0["status"] = "FAIL"
    record = {"phase0": phase0, "phase1": record_phase1(results),
              "manifest_hashes": identity["manifest_hashes"], "pixel_reads": pixel_reads,
              "tests": {"cpu": cpu, "cuda": cuda}}
    for number in range(2, 7):
        record[f"phase{number}"] = {"status": "BLOCKED", "reason": f"Phase {number - 1} has no full pass"}
    write_json(results / "acceptance.json", record)
    if (args.benchmark_dir / "benchmark.csv").exists() and args.benchmark_dir != results / "phase0":
        write_csv(results / "phase0/benchmark.csv", csv_at(args.benchmark_dir / "benchmark.csv"))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
