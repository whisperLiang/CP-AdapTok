"""Freeze experiment rules and verify the parent commit's protected assets."""

import fcntl
import hashlib
from contextlib import contextmanager

from aip_vtok.c1.artifacts import LOCKED, timestamp
from aip_vtok.c1.scan import BUDGETS, REF_BUDGET, SIGMAS
from aip_vtok.c1_followup.io import CANDIDATES, protected_checksums, sealed_controls
from aip_vtok.diagnostics.diag0.io import command, json_at, metric_weight_hashes
from aip_vtok.evaluation.fix import HORIZON_VIDEO
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json

from .scan import REFINEMENT_PCTS

BASE = "a8474ee"
BRANCH = "experiment/aip-vtok-sparse-rfsq"
OUTPUT = resolve("results/aip_vtok/sparse_rfsq")
WEIGHTS = resolve("checkpoints/aip_vtok/sparse_rfsq")
REPORT = resolve("docs/aip_vtok_sparse_rfsq_report.md")
WORK_LOG = resolve("docs/aip_vtok_sparse_rfsq_work_log.md")
PROTOCOL = OUTPUT / "protocol.json"
SCAN_LOCK = OUTPUT / "scan_lock.json"
CALIBRATION = OUTPUT / "calibration"
TEST_GATE = OUTPUT / "test_gate.json"
S1_GATE = OUTPUT / "s1_gate.json"
S2_GATE = OUTPUT / "s2_gate.json"
COHORT_FRAMES = 174
HORIZON = 64
FOCUS = (510, 2048, 8192)
PILOTS = {"W1": 0.15, "W2": 0.05}
SEED, UPDATES, DIAG_EVERY = 1, 500, 50
ANCHOR_WEIGHT, REFERENCE_WEIGHT = 0.05, 0.15
SMOKE_LENGTHS = (2, 4)
CALIBRATION_SEED, CALIBRATION_VIDEOS, CALIBRATION_LENGTH = 1701, 6, 4
MIN_POSITIVE_STEPS = 2
LIMITS = {
    "focus_min_psnr_gain_db": 0.5, "focus_min_passing_budgets": 2, "focus_max_dists_increase": 0.005,
    "late_min_psnr_gain_db": -0.10, "late_max_lpips_increase": 0.005, "late_max_dists_increase": 0.005,
    "max_reference_latent_rms": 2.0, "late_max_rms_relative_to_b2": 1.25,
    "max_psnr_backslide_floor_db": 1.0, "max_psnr_backslide_over_b2_db": 0.25,
}


def freeze_protocol(config, identity):
    data = {
        "schema": 1, "base_commit": command("git", "rev-parse", BASE), "identity": identity,
        "c1_scan_sha256": file_hash(LOCKED), "c1_followup_candidates_sha256": file_hash(CANDIDATES),
        "metric_weights": metric_weight_hashes(), "budgets": list(BUDGETS),
        "reference_budgets": {"I": REF_BUDGET[True], "P": REF_BUDGET[False]},
        "fixed_reference": "sealed C1 A1/B2 at I2048/P510, shared across all displayed budgets and methods",
        "independent_closed": "separate sender and receiver states committed at I2048/P510 with their own method",
        "calibration": {"seed": CALIBRATION_SEED, "videos": CALIBRATION_VIDEOS, "clip_frames": CALIBRATION_LENGTH,
                        "excludes": "Gate 0 calibration, C1 calibration and C1-Followup quantizer fit videos. "
                                    "S2 excludes the selected Train videos and any duplicate frame hashes.",
                        "sigma_grid": list(SIGMAS), "sp3_level2_percentages": list(REFINEMENT_PCTS),
                        "score": "Mean of ten I/P × K Train global-MSE PSNR scores on the fixed reference. "
                                 "For each sigma, average SP-1, SP-2 and the mean over SP-3 ratios. "
                                 "At that common sigma choose the best SP-3 ratio, then the best eligible schedule. "
                                 "Ties use lower sigma, lower refinement percentage and SP-1/2/3 order.",
                        "eligible": "Positive endpoint gain and at least two positive steps in the I/P-balanced "
                                    "curve. Validation vetoes failed progression or invalid closure without reselection.",
                        "minimum_positive_steps": MIN_POSITIVE_STEPS},
        "validation": config["stage1"]["validation"],
        "horizon": {"video": HORIZON_VIDEO, "frames": HORIZON, "gop": config["gop"], "late": [48, 63]},
        "rate": "16 × K payload bits or Train empirical entropy per level / group / frame type times symbols. "
                "Both estimate display-only marginal rates. Reference-maintaining prefixes and simulated framing "
                "are recorded separately.",
        "pilot": {"seed": SEED, "steps_per_weight": UPDATES, "lpips": PILOTS, "smoke_lengths": list(SMOKE_LENGTHS),
                  "loss": {**config["stage1"]["loss"], "latent_anchor": ANCHOR_WEIGHT, "reference": REFERENCE_WEIGHT},
                  "baseline": "B2", "steps_between_gradient_diagnostics": DIAG_EVERY, "focus_budgets": list(FOCUS),
                  "gate_scopes": "Quality gains on 174 frames at focus budgets. Late quality and RMS on the "
                                 "64-frame horizon at all five budgets. Max adjacent PSNR backslide in both scopes.",
                  "limits": LIMITS},
    }
    if PROTOCOL.is_file():
        data["locked_at"] = json_at(PROTOCOL)["locked_at"]
    else:
        data["locked_at"] = timestamp()
    from aip_vtok.c1_followup.io import create_record

    create_record(PROTOCOL, data)


@contextmanager
def experiment_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def allowed(path):
    return path.startswith(("aip_vtok/sparse_rfsq/", "results/aip_vtok/sparse_rfsq/",
                            "checkpoints/aip_vtok/sparse_rfsq/")) or path in {
        "scripts/aip_vtok/run_sparse_rfsq.py", "tests/aip_vtok/test_sparse_rfsq.py",
        str(REPORT.relative_to(ROOT)), str(WORK_LOG.relative_to(ROOT)),
    }


def protected_digest():
    changed = command("git", "diff", "--name-only", "--ignore-submodules=none", BASE).splitlines()
    if any(not allowed(path) for path in changed):
        raise RuntimeError("working tree modifies protected inputs from a8474ee")
    paths = {ROOT / path for path in command("git", "ls-tree", "-r", "--name-only", BASE).splitlines()}
    for own in (OUTPUT, WEIGHTS):
        paths.update(p for p in own.parent.rglob("*") if p.is_file() and own not in p.parents)
    checksums = protected_checksums(paths)
    digest = hashlib.sha256()
    for path, value in sorted(checksums.items()):
        digest.update(f"{path}\t{value}\n".encode())
    return {"sha256": digest.hexdigest(), "files": len(checksums), "controls": sealed_controls()}


def source_digest():
    paths = sorted((ROOT / "aip_vtok/sparse_rfsq").glob("*.py"))
    paths.extend((ROOT / "scripts/aip_vtok/run_sparse_rfsq.py", ROOT / "tests/aip_vtok/test_sparse_rfsq.py"))
    digest = hashlib.sha256()
    for path in paths:
        if path.name not in {"report.py", "figures.py"}:
            digest.update(f"{path.name}\t{file_hash(path)}\n".encode())
    return digest.hexdigest()


def write_integrity(assets):
    path = OUTPUT / "protected.json"
    if path.is_file():
        if assets != json_at(path):
            raise RuntimeError("protected input hashes changed since experiment creation")
    else:
        write_json(path, assets)
