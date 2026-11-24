"""Finalize phase evidence, read ledgers, and original/new asset identities."""

# ruff: noqa: E402
import argparse
import hashlib
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.training.checkpoint import read_checkpoint
from scripts.hla_tok.v2.gates import pilot_gate_path
from scripts.hla_tok.v2.protocol import CHECKPOINTS, OUTPUT, ROOT, audit_legacy, csv_at, json_at


def git_snapshot(path):
    def invoke(*args):
        return subprocess.check_output(["git", "-C", str(path), *args])

    return {
        "commit": invoke("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(invoke("diff", "--binary", "HEAD", "--")).hexdigest(),
        "status": invoke("status", "--porcelain").decode().splitlines(),
    }


def verify_legacy(legacy, original, joint):
    baseline = json_at(OUTPUT / "diag0/legacy_assets.json")
    final = audit_legacy(legacy)
    if baseline != final:
        raise RuntimeError("old checkpoints, official weights or legacy evidence changed")
    for name, repository in (("original", original), ("joint", joint),
                             ("official_source", original / "third_party/AdapTok")):
        actual = git_snapshot(repository)
        if actual != baseline["repositories"][name]:
            raise RuntimeError(f"old worktree differs from the frozen audit: {name}")
    post = OUTPUT / "diag0/replay_vq_post_refactor/validation_metrics.csv"
    if post.is_file():
        target, = csv_at(legacy / "results/hla_tok/phase1/fixed/validation_metrics.csv")
        current, = csv_at(post)
        for metric in ("global_psnr", "lpips", "lpips_vgg", "dists", "mse", "ms-ssim"):
            if not math.isclose(float(target[metric]), float(current[metric]), abs_tol=1e-8, rel_tol=1e-7):
                raise RuntimeError(f"R0 changed after the V2 architecture refactor: {metric}")
        if current["total_bytes"] != target["total_bytes"] or current["frames"] != "174":
            raise RuntimeError("R0 post-refactor rate ledger differs from the frozen measurement")
        final["post_refactor_r0_replay_verified"] = True
    final["verified_unchanged"] = True
    return final


def integrity_checkpoints():
    records = []
    for path in sorted(CHECKPOINTS.rglob("step_*.pt")):
        state = read_checkpoint(path).state
        records.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                        "step": state["step"], "data_cursor": state["data_cursor"],
                        "architecture": state["counters"].get("architecture"),
                        "identity": state["identity"]})
    write_json(OUTPUT / "checkpoint_manifest.json", records)
    return len(records)


def record_test_counts(path):
    if not path.is_file():
        return None
    cases = ET.parse(path).findall(".//testcase")
    bad = sum(item.find("failure") is not None or item.find("error") is not None for item in cases)
    skipped = sum(item.find("skipped") is not None for item in cases)
    return {"passed": len(cases) - bad - skipped, "failed": bad, "skipped": skipped}


def file_grade(path, missing):
    return json_at(path) if path.exists() else {"status": "BLOCKED", "reason": missing}


def grades():
    d0 = file_grade(OUTPUT / "diag0/diagnosis.json", "The R0 diagnosis is incomplete")
    d1 = file_grade(OUTPUT / "diag1_overfit/acceptance.json", "D0 has no full pass")
    seed1 = pilot_gate_path(OUTPUT / "continuous_pilot", 1)
    decision = OUTPUT / "continuous_pilot/selection.json"
    d2 = file_grade(decision if decision.is_file() else seed1, "D1 has no full pass")
    if d2["status"] == "PASS" and not decision.is_file():
        d2 = d2 | {"status": "PARTIAL PASS", "reason": "Seed 2, intervention and performance evidence pending"}
    return {
        "D0": {"status": d0["status"]}, "D1": d1, "D2": d2,
        "D3": file_grade(OUTPUT / "quantized/acceptance.json", "No production candidate passed D2"),
        "D4": file_grade(OUTPUT / "native/acceptance.json", "D3 has no full pass"),
        "D5": file_grade(OUTPUT / "benchmark/acceptance.json", "D4 has no full pass"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    args = parser.parse_args()
    counts = {f"{tag}_pixel_reads": 0 for tag in ("train", "validation", "test")}
    for path in OUTPUT.rglob("pixel_reads.json"):
        for key, count in json_at(path).items():
            counts[key] += count
    if counts["test_pixel_reads"]:
        raise RuntimeError("sealed Test pixels appeared in the V2 read ledger")
    write_json(OUTPUT / "legacy_assets_final.json", verify_legacy(args.legacy_root, args.original, args.joint))
    record = grades()
    record["pixel_reads"] = counts
    record["tests"] = {kind: record_test_counts(OUTPUT / f"{kind}_tests.xml") for kind in ("cpu", "cuda")}
    mutations = OUTPUT / "mutations/summary.json"
    if mutations.exists():
        record["mutations"] = json_at(mutations)
    record["verified_checkpoints"] = integrity_checkpoints()
    record["phase2_eligible"] = record["D5"]["status"] == "PASS"
    write_json(OUTPUT / "acceptance.json", record)


if __name__ == "__main__":
    main()
