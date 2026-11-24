"""Frozen frame plans and read-only legacy inputs for the V2 comparisons."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import torch

from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/hla_tok/v2"
OUTPUT = ROOT / "results/hla_tok/v2"
CHECKPOINTS = ROOT / "checkpoints/hla_tok/v2"
COHORT = ROOT / "configs/hla_tok/validation_cohort.json"
BASE = "7c158491b18fab87bea03b114a7ac242368e99b4"
FIXED_BUDGET = 16320


def json_at(path):
    return json.loads(path.read_bytes())


def csv_at(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def setup(seed=1):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False


def frozen_data():
    identity = json_at(COHORT)
    reader = AuditedFrames(ROOT / "data/manifests", identity["manifest_hashes"])
    return reader, load_cohort(COHORT, reader), identity


def heldout_examples():
    return json_at(CONFIG / "diagnostic_frames.json")["validation"]


def audit_read_counts(output, reader):
    write_json(output / "pixel_reads.json", {f"{k}_pixel_reads": v for k, v in reader.counts.items()})


def verified_checkpoints(legacy):
    records = json_at(legacy / "results/hla_tok/phase1/checkpoint_manifest.json")
    chosen = {}
    for stage, step in (("continuous", 500), ("fixed", 4500)):
        record, = (r for r in records if r["stage"] == stage and r["step"] == step)
        path = legacy / record["checkpoint"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen R0 checkpoint SHA-256 differs: {path}")
        chosen[stage] = path
    return chosen


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def audit_legacy(legacy):
    if _git(legacy, "rev-parse", "HEAD") != BASE:
        raise RuntimeError("the R0 worktree must point to the frozen V1 commit")
    prior = legacy / "results/hla_tok"
    sealed = json_at(prior / "audit/final/assets.json")
    assets = dict(sealed["sha256"])
    for row in json_at(prior / "phase1/checkpoint_manifest.json"):
        assets[str(legacy / row["checkpoint"])] = row["sha256"]
    for path in (prior / "acceptance.json", prior / "baselines/metrics.csv", prior / "phase1/validation_metrics.csv",
                 prior / "phase0/benchmark.csv", prior / "final_report.md"):
        rel = str(path.relative_to(legacy))
        git_bytes = subprocess.check_output(["git", "-C", str(legacy), "show", f"{BASE}:{rel}"])
        assets[str(path)] = hashlib.sha256(git_bytes).hexdigest()
    for path, expected in assets.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"legacy experiment asset SHA-256 differs: {path}")
    repos = {key: dict(row) for key, row in sealed["repositories"].items()}
    repos["r0"] = {"commit": BASE, "status": _git(legacy, "status", "--porcelain").splitlines(),
                   "path": str(legacy)}
    source, = [Path(p).parent for p in assets if p.endswith("/model.safetensors")]
    repos["official_checkpoint"] = str(source)
    upstream = legacy / "third_party/AdapTok"
    if _git(upstream, "rev-parse", "HEAD") != repos["official_source"]["commit"] or _git(upstream, "status", "--porcelain"):
        raise RuntimeError("official AdapTok source differs from the frozen submodule")
    return {"repositories": repos, "asset_sha256": assets}
