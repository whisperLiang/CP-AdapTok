"""Fail-closed phase transitions and the dense training pilot."""

import torch

from aip_vtok.diagnostics.diag0.io import Catalog, json_at
from aip_vtok.runtime import file_hash, write_json

from .artifacts import (
    LOCKED,
    OUTPUT,
    THRESHOLD_DB,
    audit,
    finish_audit,
    frozen_inputs,
    load_frozen,
    locked_artifacts,
    timestamp,
)
from .calibrate import load_fits
from .dense_eval import evaluate_dense
from .training import SEEDS, Run, load_model, smoke, train_seed

A_UPDATES = 1000
B_UPDATES = 1500
PHASE_A = OUTPUT / "c1a"
PHASE_B = OUTPUT / "c1b"
DIAG = OUTPUT / "diag05/decision.json"


def require_audited(result, names):
    for name in names:
        path = OUTPUT / f"{name}_audit.json"
        if path.is_file() and json_at(path)["end"] >= result["finished"]:
            return
    raise RuntimeError("stage lacks a completed sealed-artifact audit")


def phase1_record(lock_path=LOCKED, record_path=DIAG):
    if not record_path.is_file():
        raise RuntimeError("complete the 64-frame Diag-0.5 gate before C1-A")
    result = json_at(record_path)
    if (not result["completed"] or not result["state_sync"] or result["closed_frames_per_method"] != 64
            or result["locked_sha256"] != file_hash(lock_path)):
        raise RuntimeError("Diag-0.5 gate is incomplete or was measured on a different locked schedule")
    require_audited(result, ("diag05_all", "diag05_closed"))
    return result


def require_sparse_release(diag, dense, lock_hash):
    if not diag.get("passes_for_c1b"):
        raise RuntimeError("Phase 1 gate withheld permission to train C1-B")
    expected = {str(seed): A_UPDATES for seed in SEEDS}
    if (not dense.get("passes_for_c1b") or not dense.get("completed")
            or dense.get("locked_sha256") != lock_hash or dense.get("steps_completed") != expected):
        raise RuntimeError("C1-B requires a passing completed dual-seed Phase 2 on the same lock")


def train_inputs():
    config, source_identity = frozen_inputs()
    lock, artifact = locked_artifacts(source_identity)
    codec = load_frozen(config)
    train = Catalog.open(config, "train")
    return config, source_identity, lock, load_fits(artifact), codec, train


def run_a(stage):
    phase1_record()
    record = audit("c1a")
    config, source_identity, lock, fits, codec, train = train_inputs()
    loss_weights = config["stage1"]["loss"]
    protocol = PHASE_A / "protocol.json"
    if not protocol.is_file():
        write_json(protocol, {
            "locked_sha256": file_hash(LOCKED), "minimum_global_psnr_gain_db": THRESHOLD_DB,
            "histories": ["same_context_b0_fsq_history", "independent_closed"],
            "requires_lpips_improvement": True, "required_seeds": list(SEEDS), "steps_per_seed": A_UPDATES,
            "recorded_before_training": timestamp(),
        })
    for length in (2, 4):
        result = recorded_smoke(train, codec, fits, lock, "c1a", length, loss_weights)
        if not result["passed"]:
            raise RuntimeError(f"C1-A {length}-frame smoke did not improve its fixed-clip objective")
    if stage != "smoke":
        models = {}
        for seed in SEEDS:
            run = Run("c1a", seed, A_UPDATES, 4)
            identity = train_seed(codec, fits, lock, run, loss_weights, train, source_identity)
            models[seed] = load_model(run, identity)
        gains, passed = evaluate_dense(config, codec, fits, lock, models)
        write_json(PHASE_A / "decision.json",
                   {"completed": True, "passes_for_c1b": passed, "gains": gains,
                    "locked_sha256": file_hash(LOCKED), "steps_completed": {str(seed): A_UPDATES for seed in SEEDS},
                    "cohort_frames": config["stage1"]["validation"]["total_frames"],
                    "finished": timestamp()})
    finish_audit(record)


def recorded_smoke(train, codec, fits, lock, phase, length, loss_weights, start=0):
    path = OUTPUT / phase / f"smoke_{length}f.json"
    if path.is_file():
        result = json_at(path)
        if result["scan_lock_sha256"] == file_hash(LOCKED):
            return result
    return smoke(train, codec, fits, lock, Run(phase, 7302, 10, length), loss_weights, phase=start)


def sparse_length(train, codec, fits, lock, loss_weights):
    for length, phase in ((2, 0), (4, 12), (8, 26)):
        try:
            result = recorded_smoke(train, codec, fits, lock, "c1b", length, loss_weights, phase)
        except torch.OutOfMemoryError as exc:
            if length != 8:
                raise
            write_json(PHASE_B / "smoke_8f.json", {"passed": False, "reason": str(exc),
                                                  "scan_lock_sha256": file_hash(LOCKED)})
            return 4
        if not result["passed"]:
            if length == 8:
                return 4
            raise RuntimeError(f"C1-B {length}-frame smoke did not improve its fixed-clip objective")
    return 8


def run_b(stage):
    diag = phase1_record()
    path = PHASE_A / "decision.json"
    dense = json_at(path) if path.is_file() else {}
    require_sparse_release(diag, dense, file_hash(LOCKED))
    require_audited(dense, ("c1a",))
    from .sparse_eval import evaluate_sparse

    record = audit("c1b")
    config, identity, lock, fits, codec, train = train_inputs()
    loss_weights = config["stage1"]["loss"]
    length = sparse_length(train, codec, fits, lock, loss_weights)
    smoke_record = json_at(PHASE_B / f"smoke_{length}f.json")
    write_json(PHASE_B / "rollout.json", {
        "training_window": length, "rollout": smoke_record["rollout"], "smoke_seconds_per_step":
        smoke_record["median_seconds_per_update"], "smoke_peak_allocated_mib": smoke_record["peak_allocated_mib"],
        "estimated_dual_seed_update_hours": smoke_record["median_seconds_per_update"] * 2 * B_UPDATES / 3600,
        "target_steps_per_seed": B_UPDATES, "release_diag05": file_hash(DIAG), "release_c1a": file_hash(path),
    })
    if stage != "smoke":
        models = {}
        for seed in SEEDS:
            run = Run("c1b", seed, B_UPDATES, length, late_gop=True)
            inputs = train_seed(codec, fits, lock, run, loss_weights, train, identity)
            models[seed] = load_model(run, inputs)
        paired, closed = evaluate_sparse(config, codec, fits, lock, models)
        write_json(PHASE_B / "decision.json",
                   {"completed": True, "locked_sha256": file_hash(LOCKED), "rollout": smoke_record["rollout"],
                    "steps_completed": {str(seed): B_UPDATES for seed in SEEDS}, "matched_gains": paired,
                    "closed_gains": closed, "finished": timestamp()})
    finish_audit(record)
