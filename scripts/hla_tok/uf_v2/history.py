"""Verify read-only historical worktrees, checkpoint ledgers, and byte evidence."""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from hla_tok.evaluation.data import load_manifests
from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import OfficialAssets, sha256_file
from scripts.hla_tok.uf.evidence import archive_curves
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

UF = "3efe25ea47308cd7786888f0f5adbd0eb449e64d"
CC = "a483a212d8efeaeab045daf4f0f5d0830d69928f"
COMMON = "502eaf64f6fe8693dc90ddf6164c87799f1aebfe"
BRANCH = "refs/heads/experiment/hla-tok-uf-v2"


def git(path: Path, *command: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *command], text=True).rstrip("\n")


def tree(path: Path, commit: str) -> dict[str, str]:
    records = git(path, "ls-tree", "-rz", commit).strip("\0").split("\0")
    return {name: meta.split()[2] if meta.split()[1] == "commit" else sha256_file(path / name)
            for meta, name in (record.split("\t", 1) for record in records)}


def original(root: Path, cohort: str) -> dict:
    folder = root / "results/hla_tok/compact"
    identity = json_at(folder / "original_archive_identity.json")
    if identity["cohort_sha256"] != cohort:
        raise ValueError("Original AdapTok has a mismatched cohort")
    by_budget: dict[int, list[dict]] = defaultdict(list)
    for window in csv_at(folder / "original_window_ledger.csv"):
        if int(window["total_bytes"]) != int(window["header_bytes"]) + int(window["payload_bytes"]):
            raise ValueError("Original window byte ledger mismatch")
        by_budget[int(window["budget_per_group"])].append(window)
    for row in csv_at(folder / "original_rd.csv"):
        windows = by_budget[int(row["budget_per_group"])]
        if sum(int(w["valid_frames"]) for w in windows) != 174:
            raise ValueError("Original frame count differs")
        if sum(int(w["total_bytes"]) for w in windows) != int(row["total_bytes"]):
            raise ValueError("Original rate differs from packed F13W window bytes")
    archive = root / "results/hla_tok/baselines"
    for name, digest in identity["archive_sha256"].items():
        if sha256_file(archive / "stride128_bf16" / name) != digest:
            raise ValueError(f"Original replay source differs: {name}")
    if sha256_file(archive / "official_weights.json") != identity["archive_weights_sha256"]:
        raise ValueError("Original weight archive differs")
    if git(root, "rev-parse", "experiment/original-adaptok-1080p") != identity["archive_commit"]:
        raise ValueError("Original archival reference differs")
    return identity


def checkpoints(uf: Path, cc: Path) -> dict:
    directory = uf / "checkpoints/hla_tok/uf/isolated"
    records = {}
    for row in csv_at(uf / "results/hla_tok/uf/evidence/checkpoint_ledger.csv"):
        path = directory / Path(row["checkpoint"]).name
        current = sha256_file(path)
        if current != row["sha256"]:
            raise ValueError(f"UF checkpoint digest differs: {path}")
        records[str(path)] = current
    n2 = json_at(cc / "results/hla_tok/cc/training/cc-n-seed1-step2000.json")
    for key, digest in (("checkpoint", "checkpoint_sha256"), ("model_asset", "exported_asset_sha256")):
        path = Path(n2[key])
        observed = sha256_file(path)
        if observed != n2[digest]:
            raise ValueError(f"CC-N N2 digest differs: {path}")
        records[str(path)] = observed
    return records


def manifest(uf: Path, cc: Path, weights: Path) -> dict:
    if git(uf, "rev-parse", "HEAD") != UF or git(cc, "rev-parse", "HEAD") != CC:
        raise ValueError("historical worktrees have moved")
    if git(ROOT, "merge-base", UF, CC) != COMMON:
        raise ValueError("historical merge base differs")
    assets = OfficialAssets.verify(weights)
    refs = dict(row.split() for row in git(ROOT, "for-each-ref", "--format=%(refname) %(objectname)",
                                          "refs/heads").splitlines())
    refs.pop(BRANCH, None)
    return {"merge_base": COMMON, "refs": refs, "uf": {"commit": UF, "files": tree(uf, UF)},
            "cc": {"commit": CC, "files": tree(cc, CC)}, "v2_inherited": tree(ROOT, UF),
            "checkpoints": checkpoints(uf, cc) | {str(path): sha256_file(path)
                                                 for path in (assets.intra, assets.ld)}}


def baseline(uf: Path, cc: Path, state: dict) -> dict:
    uf_results, cc_results = uf / "results/hla_tok/uf", cc / "results/hla_tok/cc"
    quality, original_scope = archive_curves()
    protocol = json_at(cc_results / "evidence/shared_protocol.json")
    identity = original(ROOT, protocol["cohort_sha256"])
    experiment = json_at(cc_results / "evaluation/results.json")
    _, manifest_hashes = load_manifests(ROOT / "data/manifests")
    if manifest_hashes != protocol["frozen_manifest_sha256"]:
        raise ValueError("canonical frozen manifest digests differ")
    for file, digest in protocol["evaluation_code_sha256"].items():
        if sha256_file(cc / file) != digest:
            raise ValueError(f"historical evaluator drift for {file}")
    for file, identity_row in protocol["identities"]["scheme_b"]["artifacts"].items():
        if sha256_file(uf / identity_row["file"]) != identity_row["sha256"]:
            raise ValueError(f"historical UF evidence drift for {file}")
    return {"status": "PASS", "independent_fresh_measurement": False, "original": identity | original_scope,
            "refs": {"UF": UF, "CC": CC, "common": COMMON}, "shared_protocol": protocol,
            "curve_sources": {
                "original_and_temporal": quality,
                "cc": {name: experiment["quality_curves"][name] for name in ("CC-S", "CC-N", "CC-J")},
                "rt": json_at(cc_results / "baseline/DCVC-RT/summary.json"),
                "uf_ld": json_at(uf_results / "baseline/isolated/validation.json"),
                "uf_anchor": json_at(uf_results / "anchor/A2-validation-bf16/summary.json"),
                "uf_anchor_seed2": json_at(uf_results / "anchor/A2-seed2-validation-bf16/summary.json"),
                "uf_d0": json_at(uf_results / "progressive/D0-validation/summary.json"),
            },
            "original_equal_quality": [row for row in experiment["equal_quality"]
                                       if row["method"] == "Original AdapTok"],
            "asset_counts": {name: len(state[name]["files"]) for name in ("uf", "cc")},
            "test_pixel_reads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("snapshot", "verify"))
    parser.add_argument("--uf", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = manifest(args.uf, args.cc, args.weights)
    path = args.output / "historical_sha256.json"
    if args.stage == "snapshot" and not path.exists():
        write_json(args.output / "baseline_audit.json", baseline(args.uf, args.cc, observed))
        write_json(path, observed)
    if json.loads(path.read_text()) != observed:
        raise RuntimeError("historical assets, inherited files or protected refs changed")
    print("PASS: historical assets and protected refs match")


if __name__ == "__main__":
    main()
