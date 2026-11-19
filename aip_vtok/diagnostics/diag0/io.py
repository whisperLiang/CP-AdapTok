"""Diag-0 inputs, artifact isolation and machine-readable records."""

import csv
import fcntl
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.training.data import read_window, sources

BASE = "caae55ba0fec780c6f32bee6b9c4fa1ef5998421"
OUTPUT = resolve("results/aip_vtok/diag0")
GATE = resolve("results/aip_vtok")
FIX = GATE / "gate1_fix/fix_a/seed1"
PROJECTOR = resolve("checkpoints/aip_vtok/projector_w128/latest.pt")
FIX_WEIGHTS = resolve("checkpoints/aip_vtok/gate1_fix/fix_a/seed1/step_1000.pt")
PROBES = {
    "A": "d0_a_pca_and_sensitivity",
    "B": "d0_b_spatial_vs_channel",
    "C": "d0_c_oracle_free_topk",
    "D": "d0_d_oracle_depth_prefix",
    "E": "d0_e_closed_loop_64f",
}
READABLE_ROLES = {"train": {"train"}, "validation": {"validation", "historical-development"}}


def json_at(path):
    return json.loads(path.read_text())


def command(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


@dataclass(frozen=True)
class Catalog:
    split: str
    rows: dict
    manifest_hashes: dict

    @classmethod
    def open(cls, config, split):
        if split not in READABLE_ROLES:
            raise ValueError("only train and validation are pixel-readable")
        rows, hashes = sources(config, split)
        return cls(split, {row["content_id"]: row for row in rows}, hashes)

    def read(self, video, start, length):
        row = self.rows[video]
        if row["role"] not in READABLE_ROLES.get(self.split, set()):
            raise ValueError("frame request outside the declared split")
        return read_window(row, start, length)


def sealed_projector():
    path = str(PROJECTOR.relative_to(ROOT))
    expected = json_at(GATE / "checkpoints_manifest.json")["projector_w128/latest.pt"]["sha256"]
    actual = file_hash(PROJECTOR)
    if actual != expected:
        raise RuntimeError("Gate 0 projector does not match the sealed SHA256")
    return {"path": path, "sha256": actual}


def protected_artifacts():
    paths = [*resolve("checkpoints/aip_vtok").rglob("*"), *GATE.rglob("*")]
    files = sorted(p for p in paths if p.is_file() and OUTPUT not in p.parents)
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(ROOT)}\t{file_hash(path)}\n".encode())
    return {"files": len(files), "sha256": digest.hexdigest()}


def protected_code_changes():
    changes = command("git", "diff", "--name-only", BASE).splitlines()
    allow = ("aip_vtok/diagnostics/diag0/", "results/aip_vtok/diag0/")
    files = {"scripts/aip_vtok/run_diag0.py", "tests/aip_vtok/test_diag0.py", "docs/aip_vtok_diag0_report.md"}
    return [path for path in changes if path not in files and not path.startswith(allow)]


def gpu_snapshot():
    gpu = command(
        "nvidia-smi", "-i", "0", "--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu",
        "--format=csv,noheader",
    )
    processes = command("nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader")
    foreign = [line for line in processes.splitlines() if line.strip() != str(os.getpid())]
    if "H100" not in gpu or foreign:
        raise RuntimeError(f"requires an idle H100 GPU 0; observed {gpu}; competing PIDs {foreign}")
    return {"gpu": gpu, "other_compute_pids": foreign}


@contextmanager
def run_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_probe(probe, rows, summary, **details):
    base = OUTPUT / PROBES[probe]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = base.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(base.with_suffix(".csv"))
    write_json(base.with_suffix(".json"), {"summary": summary, **details, "rows": rows})


def read_probe(probe):
    return json_at(OUTPUT / f"{PROBES[probe]}.json")


def native_control():
    source = GATE / "native_bitstream_control.json"
    original = json_at(source)
    pixels = 1080 * 1920
    total = sum(row["total_bytes"] for row in original["rows"])
    bpp = total * 8 / (pixels * len(original["rows"]))
    if not original["real_bitstream"] or bpp != original["bpp"] or original["summary"]["frames"] != 64:
        raise ValueError("invalid sealed DMC-LD SPS/IP byte ledger")
    return {
        "video": "DAVIS2017:car-roundabout", "gop": 32, "qp": 32, "real_bitstream": True,
        "bpp": bpp, "kbit_per_frame": bpp * pixels / 1000, "total_bytes": total,
        "summary": original["summary"], "source": str(source.relative_to(ROOT)), "sha256": file_hash(source),
    }


def metric_weight_hashes():
    import DISTS_pytorch
    import lpips
    import torch

    hub = Path(torch.hub.get_dir()) / "checkpoints"
    paths = (
        hub / "alexnet-owt-7be5be79.pth", hub / "vgg16-397923af.pth",
        Path(lpips.__file__).with_name("weights") / "v0.1/alex.pth",
        Path(DISTS_pytorch.__file__).with_name("weights.pt"),
    )
    labels = ("imagenet_alexnet", "imagenet_vgg16", "lpips_alex_v01", "dists")
    return {label: file_hash(path) for label, path in zip(labels, paths)}
