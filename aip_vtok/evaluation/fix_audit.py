"""Byte-level checks of the pre-fix evidence and protected code."""

import subprocess

from aip_vtok.runtime import ROOT, file_hash, write_json
from aip_vtok.training.fix import CHECKPOINTS, RESULTS, read_json

BASE = "508a97e"
PROTECTED = (
    "third_party",
    "cp_adaptok",
    "adaptok_hr",
    "configs/aip_vtok/gate0.yaml",
    "aip_vtok/backbones",
    "aip_vtok/decoder",
    "aip_vtok/training/trainer.py",
    "aip_vtok/evaluation/gate0.py",
    "aip_vtok/evaluation/evaluate.py",
    "scripts/aip_vtok/run_gate0.sh",
)


def protected_hashes():
    files = [ROOT / "checkpoints" / f"cvpr2026_{name}.pth.tar" for name in ("image", "video_ld")]
    for new_root in (RESULTS, CHECKPOINTS):
        files.extend(p for p in new_root.parent.rglob("*") if p.is_file() and new_root not in p.parents)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def audit():
    baseline_file = RESULTS / "integrity_baseline.json"
    current = protected_hashes()
    if not baseline_file.exists():
        write_json(baseline_file, current)
    baseline = read_json(baseline_file)
    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", *PROTECTED],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    modified = sorted(path for path, expected in baseline.items() if current.get(path) != expected)
    source = "checkpoints/aip_vtok/projector_w128/latest.pt"
    manifest_sha = read_json(RESULTS.parent / "checkpoints_manifest.json")["projector_w128/latest.pt"][
        "sha256"
    ]
    result = {
        "base_commit": BASE,
        "protected_file_count": len(baseline),
        "protected_code_diff": diff,
        "modified_or_missing": modified,
        "projector_w128": {
            "path": source,
            "before": baseline[source],
            "after": current.get(source),
            "sealed": manifest_sha,
        },
        "passed": not diff and not modified and current.get(source) == manifest_sha,
    }
    write_json(RESULTS / "integrity.json", result)
    if not result["passed"]:
        raise RuntimeError("protected Gate 0 or historical artifacts changed")
    return result


def checkpoint_manifest():
    weights = {
        str(path.relative_to(CHECKPOINTS)): {"sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(CHECKPOINTS.rglob("*.pt"))
    }
    write_json(RESULTS / "checkpoints_manifest.json", weights)


if __name__ == "__main__":
    print(audit())
    checkpoint_manifest()
