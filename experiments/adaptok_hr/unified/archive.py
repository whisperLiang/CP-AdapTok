"""Portable real evidence with streaming per-member digest verification."""

import argparse
import hashlib
import json
import shutil
import tarfile

from adaptok_hr.runtime import file_hash, write_json

from .run import CHECKPOINTS, PACKAGE, ROOT

EVIDENCE = PACKAGE / "evidence"
SUFFIXES = {".json", ".jsonl", ".yaml", ".log", ".xml", ".bin", ".md"}


def selected_native_images():
    decision = json.loads((ROOT / "decision_A.json").read_text())
    winner = decision["winner"]
    names = [(f"S{s}_{winner}", 1600) for s in (1, 2)]
    milestones = sorted(int(p.stem.split("_")[-1]) for p in ROOT.glob("decision_B_*.json"))
    if milestones:
        step = milestones[-1]
        for seed in (1, 2):
            candidates = []
            for arm in ("N0", "N1", "N2"):
                name = f"S{seed}_{arm}"
                summary = json.loads((ROOT / name / f"validation_{step}_64/summary.json").read_text())
                candidates.append((summary["closed_loop"]["2048"]["lpips"], name))
            names.append((min(candidates)[1], step))
    for name, step in names:
        target = EVIDENCE / "native_samples" / name
        target.mkdir(parents=True, exist_ok=True)
        source = ROOT / name / f"validation_{step}_64/samples"
        for path in source.glob("car-roundabout_0031_*.png"):
            shutil.copy2(path, target / path.name)


def archive():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix in SUFFIXES and "figures" not in p.parts)
    paths += sorted(CHECKPOINTS.rglob("*.json"))
    manifest = {str(p): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in paths}
    path = EVIDENCE / "raw_records.tar.gz"
    with tarfile.open(path.with_suffix(".tmp"), "w:gz", compresslevel=6) as handle:
        for source in paths:
            handle.add(source, arcname=str(source), recursive=False)
    path.with_suffix(".tmp").replace(path)
    write_json(EVIDENCE / "manifest.json", {"archive": path.name, "archive_sha256": file_hash(path), "members": manifest})
    for name in ("decision_A.json", "calibration.json", "acceptance.json", "mutations.json", "quality_audit.json"):
        shutil.copy2(ROOT / name, EVIDENCE / name)
    for source in ROOT.glob("decision_B_*.json"):
        shutil.copy2(source, EVIDENCE / source.name)
    shutil.copytree(ROOT / "configs", PACKAGE / "configs", dirs_exist_ok=True)
    if (ROOT / "figures").exists():
        shutil.copytree(ROOT / "figures", EVIDENCE / "figures", dirs_exist_ok=True)
    selected_native_images()
    verify()


def verify():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    path = EVIDENCE / manifest["archive"]
    if file_hash(path) != manifest["archive_sha256"]:
        raise ValueError("archive digest mismatch")
    checked = set()
    with tarfile.open(path, "r:gz") as handle:
        for member in handle:
            expected = manifest["members"][member.name]
            digest = hashlib.sha256()
            stream = handle.extractfile(member)
            for chunk in iter(lambda: stream.read(2**20), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected["sha256"] or member.size != expected["bytes"]:
                raise ValueError("evidence member digest mismatch")
            checked.add(member.name)
    if checked != set(manifest["members"]):
        raise ValueError("evidence member set mismatch")
    print(json.dumps({"verified_members": len(checked), "archive_bytes": path.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else archive()
