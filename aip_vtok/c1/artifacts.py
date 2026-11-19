"""Read the sealed controls and write only to the C1 experiment directories."""

import csv
import fcntl
import hashlib
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

from aip_vtok.diagnostics.diag0.io import (
    FIX,
    FIX_WEIGHTS,
    GATE,
    command,
    gpu_snapshot,
    json_at,
    sealed_projector,
)
from aip_vtok.diagnostics.diag0.io import (
    OUTPUT as DIAG0,
)
from aip_vtok.diagnostics.diag0.metrics import summarize
from aip_vtok.evaluation.fix import load_evaluation_codec
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.fix import fix_settings

BASE = "3cb3641"
OUTPUT = resolve("results/aip_vtok/c1")
WEIGHTS = resolve("checkpoints/aip_vtok/c1")
LOCKED = OUTPUT / "scan_lock.json"
FITTED = WEIGHTS / "train_fit.pt"
THRESHOLD_DB = 0.5


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def frozen_inputs():
    config = fix_settings("fix_a", 1)
    history = json_at(DIAG0 / "provenance.json")
    identity = {
        "base_commit": command("git", "rev-parse", BASE),
        "official_dcvc": command("git", "-C", "third_party/DCVC", "rev-parse", "HEAD"),
        "manifests": history["manifests_metadata_hashes"],
        "alignment": file_hash(resolve(config["checkpoints"]) / "alignment.pt"),
        "projector": sealed_projector()["sha256"],
        "fix_a": file_hash(FIX_WEIGHTS),
        "dmci": file_hash(config["intra_checkpoint"]),
        "dmc_ld": file_hash(config["ld_checkpoint"]),
        "qp": config["qp"],
        "gop": config["gop"],
    }
    expected = history["identity"]
    for name in ("alignment", "projector", "fix_a", "dmci", "dmc_ld"):
        source = {"dmci": "intra", "dmc_ld": "ld"}.get(name, name)
        if identity[name] != expected[f"{source}_sha256"]:
            raise RuntimeError(f"{name} differs from the sealed Diag-0 input")
    return config, identity


def load_frozen(config):
    return load_evaluation_codec(config, FIX, FIX_WEIGHTS, 1000).eval().requires_grad_(False)


def protected_digest():
    roots = (GATE, WEIGHTS.parent)
    digest = hashlib.sha256()
    count = 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and OUTPUT not in path.parents and WEIGHTS not in path.parents:
                digest.update(f"{path.relative_to(ROOT)}\t{file_hash(path)}\n".encode())
                count += 1
    return {"sha256": digest.hexdigest(), "files": count}


@contextmanager
def experiment():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def audit(stage):
    start = gpu_snapshot()
    configure_determinism()
    protected = protected_digest()
    return {"stage": stage, "start": timestamp(), "gpu": start, "protected": protected,
            "projector_before": sealed_projector(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "test_images_read": 0, "tf32": False}


def finish_audit(record):
    record.update(end=timestamp(), protected_after=protected_digest(), projector_after=sealed_projector())
    if record["protected"] != record["protected_after"] or record["projector_before"] != record["projector_after"]:
        raise RuntimeError("sealed artifacts changed during C1")
    write_json(OUTPUT / f'{record["stage"]}_audit.json', record)


def group_summary(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**summarize(group), **dict(zip(keys, identity))} for identity, group in groups.items()]


def write_rows(name, rows, keys, **details):
    base = OUTPUT / name
    base.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    tmp = base.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, base.with_suffix(".csv"))
    write_json(base.with_suffix(".json"), {"summary": group_summary(rows, keys), "rows": rows, **details})


def locked_artifacts(identity):
    locked = json_at(LOCKED)
    if locked["identity"] != identity or locked["fit_sha256"] != file_hash(FITTED):
        raise RuntimeError("C1 schedule does not match the frozen fits and source identity")
    artifact = torch.load(FITTED, map_location="cuda:0", weights_only=True)
    if artifact["identity"] != identity:
        raise RuntimeError("C1 training statistics have a different source identity")
    return locked, artifact
