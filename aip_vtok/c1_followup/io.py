"""Follow-up artifact boundaries, protected inputs and auditable dataset access."""

import csv
import fcntl
import hashlib
import os
from contextlib import contextmanager

import torch

from aip_vtok.c1.artifacts import FITTED, LOCKED
from aip_vtok.c1.artifacts import OUTPUT as LEGACY
from aip_vtok.diagnostics.diag0.io import Catalog, command, json_at, sealed_projector
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.training.data import prefetched_windows

BASE = "a0f5eb82bf138ac9ed8fa44413261c6a8dfe9dd6"
OUTPUT = resolve("results/aip_vtok/c1_followup")
WEIGHTS = resolve("checkpoints/aip_vtok/c1_followup")
REPORT = resolve("docs/aip_vtok_c1_followup_report.md")
WORK_LOG = resolve("docs/aip_vtok_c1_followup_work_log.md")
LEGACY_VALIDATION = LEGACY / "c1a/validation.json"
PROTOCOL = OUTPUT / "protocol.json"
CANDIDATES = OUTPUT / "candidates.json"
INTEGRITY = OUTPUT / "protected.json"


@contextmanager
def experiment_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def create_record(path, data):
    """An immutable record under the experiment's single writer lock."""
    if path.is_file():
        if json_at(path) != data:
            raise RuntimeError(f"frozen record differs: {path}")
    else:
        write_json(path, data)


def write_rows(path, rows, **details):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    tmp = path.with_suffix(".csv.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path.with_suffix(".csv"))
    write_json(path.with_suffix(".json"), {**details, "rows": rows})


def sealed_controls():
    legacy = json_at(LEGACY / "verification.json")
    c1b_steps = {f"seed{seed}": legacy["training_steps"][f"c1b_seed{seed}"] for seed in (1, 2)}
    if any(c1b_steps.values()) or (WEIGHTS.parent / "c1/c1b").exists():
        raise RuntimeError("legacy C1-B must remain sealed at zero steps")
    return {
        "projector": sealed_projector(), "legacy_c1b_steps": c1b_steps,
        "legacy_c1b_status": legacy["c1b_status"],
        "old_fit_sha256": file_hash(FITTED), "old_scan_lock_sha256": file_hash(LOCKED),
        "old_validation_sha256": file_hash(LEGACY_VALIDATION),
    }


def followup_path(path):
    return path.startswith((
        "aip_vtok/c1_followup/", "results/aip_vtok/c1_followup/", "checkpoints/aip_vtok/c1_followup/",
    )) or path in {
        "scripts/aip_vtok/run_c1_followup.py", "tests/aip_vtok/test_c1_followup.py",
        str(REPORT.relative_to(ROOT)), str(WORK_LOG.relative_to(ROOT)),
    }


def protected_paths():
    changed = command("git", "diff", "--name-only", "--ignore-submodules=none", BASE).splitlines()
    if any(not followup_path(path) for path in changed):
        raise RuntimeError("working tree modifies protected C1 files")
    paths = {ROOT / path for path in command("git", "ls-tree", "-r", "--name-only", BASE).splitlines()}
    for parent, own in ((OUTPUT.parent, OUTPUT), (WEIGHTS.parent, WEIGHTS)):
        paths.update(path for path in parent.rglob("*") if path.is_file() and own not in path.parents)
    return paths


def protected_checksums(paths):
    checksums = {}
    for path in sorted(paths):
        name = path.relative_to(ROOT).as_posix()
        if path.is_dir():  # Gitlink. Include the checked-out commit and actual tracked file bytes.
            checksums[name] = command("git", "-C", str(path), "rev-parse", "HEAD")
            for member in command("git", "-C", str(path), "ls-files").splitlines():
                checksums[f"{name}/{member}"] = protected_file_hash(path / member)
        else:
            checksums[name] = protected_file_hash(path)
    return checksums


def protected_digest():
    checksums = protected_checksums(protected_paths())
    digest = hashlib.sha256()
    for name, sha in sorted(checksums.items()):
        digest.update(f"{name}\t{sha}\n".encode())
    return {"sha256": digest.hexdigest(), "files": len(checksums), "controls": sealed_controls()}


def protected_file_hash(path):
    return hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else file_hash(path)


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(f"{name}\t{tensor.dtype}\t{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy())
    return digest.hexdigest()


class ReadAudit:
    def __init__(self, config, *, validation_locked=False, manifest_hashes=None):
        self.config = config
        self.validation_locked = validation_locked
        self.manifest_hashes = manifest_hashes
        self.counts = {"train": 0, "validation": 0, "test": 0}
        self.denied = 0

    def check_split(self, split):
        if split == "train" or (split == "validation" and self.validation_locked):
            return
        self.denied += 1
        raise ValueError("pixel access requires Train or a locked Validation protocol")

    def open(self, split):
        self.check_split(split)
        catalog = Catalog.open(self.config, split)
        if self.manifest_hashes is not None and catalog.manifest_hashes != self.manifest_hashes:
            raise RuntimeError("catalog manifests differ from the sealed experiment")
        return catalog

    def read(self, catalog, video, start, length):
        self.check_split(catalog.split)
        frames = catalog.read(video, start, length)
        self.counts[catalog.split] += len(frames)
        return frames

    def training_windows(self, catalog, plans):
        if catalog.split != "train" or any(row["role"] != "train" for row in catalog.rows.values()):
            raise ValueError("training pixel access is confined to Train")
        rows = [catalog.rows[video] for video in sorted(catalog.rows)]
        for clip in prefetched_windows(rows, plans):
            self.counts["train"] += len(clip)
            yield clip
