"""Snapshot and verify protected worktree files, gitlinks, and local refs."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.uf.inputs import BASELINE, OUTPUT
from scripts.hla_tok.v2.protocol import ROOT, json_at

BRANCH = "refs/heads/experiment/hla-tok-uf"


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _gitlink(name: str) -> str:
    mode, commit, stage = _git("ls-files", "--stage", "--", name).split("\t", 1)[0].split()
    if (mode, stage) != ("160000", "0"):
        raise RuntimeError(f"protected path is no longer a gitlink: {name}")
    submodule = ROOT / name
    if (submodule / ".git").exists():
        observed = _git("-C", str(submodule), "rev-parse", "HEAD")
        if observed != commit or _git("-C", str(submodule), "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"protected gitlink checkout is not clean: {name}")
    return commit


def protected_tree() -> dict:
    files, links = {}, {}
    records = _git("ls-tree", "-rz", BASELINE).rstrip("\0").split("\0")
    for entry in records:
        meta, name = entry.split("\t", 1)
        if " commit " in meta:
            links[name] = _gitlink(name)
        else:
            path = ROOT / name
            if path.is_symlink():
                files[name] = hashlib.sha256(str(path.readlink()).encode()).hexdigest()
            else:
                with path.open("rb") as source:
                    files[name] = hashlib.file_digest(source, "sha256").hexdigest()
    refs = dict(row.split() for row in _git("for-each-ref", "--format=%(refname) %(objectname)",
                                            "refs/heads").splitlines())
    refs.pop(BRANCH, None)
    return {"baseline": BASELINE, "tracked_sha256": files, "gitlinks": links, "protected_refs": refs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("snapshot", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT / "evidence")
    args = parser.parse_args()
    manifest = args.output / "historical_sha256.json"
    observed = protected_tree()
    if args.stage == "snapshot" and not manifest.exists():
        if _git("rev-parse", "HEAD") != BASELINE or _git("diff", "HEAD", "--"):
            raise RuntimeError("historical snapshot requires an unchanged baseline checkout")
        write_json(manifest, observed)
    expected = json_at(manifest)
    match = observed == expected
    write_json(args.output / "historical_audit.json", {
        "status": "PASS" if match else "FAIL", "baseline": BASELINE,
        "files": len(observed["tracked_sha256"]), "branches": len(observed["protected_refs"]),
    })
    if not match:
        raise RuntimeError("a protected historical file, submodule, or branch changed")


if __name__ == "__main__":
    main()
