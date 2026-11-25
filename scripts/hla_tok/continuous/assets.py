"""Audit read-only V3 assets without opening data pixels."""

import argparse
import subprocess
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from scripts.hla_tok.continuous.inputs import CONFIG, RESULTS, V3
from scripts.hla_tok.v2.protocol import ROOT, json_at
from scripts.hla_tok.v3.protocol import read_config

V3_TREES = ("results/hla_tok/v3", "checkpoints/hla_tok/v3", "docs/hla_tok/v3", "configs/hla_tok/v3")
UNCHANGED = ("results", "docs", "configs", "data/manifests", "third_party")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot(v3: Path) -> dict:
    base = read_config(CONFIG)["base_commit"]
    if git(v3, "rev-parse", "HEAD") != base or git(v3, "status", "--porcelain"):
        raise ValueError("the V3 source worktree differs from its clean frozen base")
    excluded = tuple(f":(exclude){tree}/hla_tok/continuous" for tree in ("results", "docs", "configs"))
    if git(ROOT, "diff", base, "--", *UNCHANGED, *excluded):
        raise ValueError("the experiment branch modified frozen historical assets")
    files = {str(path.relative_to(v3)): sha256_file(path)
             for tree in V3_TREES for path in sorted((v3 / tree).rglob("*")) if path.is_file()}
    return {"v3_head": base, "historical_sha256": files, "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("before", "after"))
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    record = snapshot(args.v3)
    reference = args.output / "historical_assets_before.json"
    if args.action == "before":
        if reference.exists():
            raise FileExistsError(reference)
        write_json(reference, record)
    else:
        if record != json_at(reference):
            raise ValueError("the external V3 assets changed during the experiment")
        write_json(args.output / "historical_assets_audit.json", record | {"status": "PASS"})
    print(f"Audited {record['files']} V3 files at {record['v3_head']}", flush=True)


if __name__ == "__main__":
    main()
