"""Single-writer C1-Followup orchestration and reproducible stage boundaries."""

import sys
from dataclasses import dataclass

import torch

from aip_vtok.c1.artifacts import frozen_inputs, load_frozen, timestamp
from aip_vtok.c1.dense_eval import validation_videos
from aip_vtok.c1.metrics import metric_suite
from aip_vtok.diagnostics.diag0.io import gpu_snapshot, json_at, metric_weight_hashes
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.training.determinism import configure_determinism

from .calibration import calibrate, legacy_inputs, sealed_training_plans
from .dense import (
    ARMS,
    Histograms,
    Layout,
    check_replay,
    evaluate_samples,
    frame_key,
    shared_frames,
    summarize_arms,
)
from .gates import LIMITS, release_dense
from .io import (
    BASE,
    CANDIDATES,
    INTEGRITY,
    LEGACY_VALIDATION,
    OUTPUT,
    PROTOCOL,
    ReadAudit,
    create_record,
    experiment_lock,
    protected_digest,
    tensor_digest,
    write_rows,
)

REPLAY = OUTPUT / "baseline.json"
VALIDATION = OUTPUT / "validation.json"
PHASE1 = OUTPUT / "phase1_gate.json"


def freeze_protocol(config, identity):
    data = {
        "schema": 1, "base": BASE, "identity": identity, "validation": config["stage1"]["validation"],
        "limits": LIMITS, "recovery_loss": config["stage1"]["loss"], "metric_weights": metric_weight_hashes(),
        "calibration": "Longest 2–4 frame unused run per Train video, disjoint by image SHA256 from both sealed "
        "1000-step plans and C1 fit; skip Gate 0 calibration videos.",
        "phase1_history": "sealed A1 quantized B0 context, shared by each arm, zero fill",
        "phase1_selection": "one A2 slot and one A3/A5-fsq representative slot; Train selects and Validation can veto",
        "continuous_rate": "hypothetical FP32 values with the same frame and per-level packet accounting",
        "sparse_label": "c1b_followup",
    }
    recorded = json_at(PROTOCOL)["locked_at"] if PROTOCOL.is_file() else timestamp()
    create_record(PROTOCOL, {**data, "locked_at": recorded})


def frozen_candidates():
    lock = json_at(CANDIDATES)
    path = OUTPUT / "calibration.json"
    if lock["protocol_sha256"] != file_hash(PROTOCOL) or lock["calibration_sha256"] != file_hash(path):
        raise RuntimeError("candidate lock and Train evidence disagree")
    calibration = json_at(path)
    if lock["shortlist"] != calibration["shortlist"] or lock["arms"] != calibration["arms"]:
        raise RuntimeError("candidate lock changed after the Train-only fit")
    if lock["validation_pixels_read_before_lock"] != 0 or lock["test_images_read"] != 0:
        raise RuntimeError("candidate lock is not Train-only")
    return lock


def load_layouts(lock, device):
    return {arm: {kind: Layout.load(case, device) for kind, case in layouts.items()}
            for arm, layouts in lock["arms"].items()}


@dataclass
class Experiment:
    config: dict
    identity: dict
    codec: object
    suite: object
    reader: ReadAudit
    frozen_before: dict
    assets_before: dict

    def frozen_sha(self):
        return {name: tensor_digest(getattr(self.codec, name)) for name in ("backbone", "projector")}

    def verify_integrity(self):
        after = self.frozen_sha()
        assets = protected_digest()
        grads = [name for name, model in (("backbone", self.codec.backbone), ("projector", self.codec.projector))
                 for parameter in model.parameters() if parameter.requires_grad or parameter.grad is not None]
        passed = assets == self.assets_before and after == self.frozen_before and not grads and self.reader.counts["test"] == 0
        result = {
            "passed": passed, "assets_before": self.assets_before, "assets_after": assets,
            "tensors_before": self.frozen_before, "tensors_after": after,
            "frozen_parameter_gradients": grads, "test_images_read": self.reader.counts["test"],
        }
        if not passed:
            raise RuntimeError(f"protected C1 assets, tensors, gradients or split access changed: {result}")
        return result


def start_experiment(assets):
    configure_determinism()
    config, identity = frozen_inputs()
    freeze_protocol(config, identity)
    codec = load_frozen(config)
    suite = metric_suite()
    reader = ReadAudit(config, manifest_hashes=identity["manifests"])
    run = Experiment(config, identity, codec, suite, reader, {}, assets)
    run.frozen_before = run.frozen_sha()
    return run


def run_calibration(run):
    if CANDIDATES.is_file():
        return frozen_candidates()
    lock, _, baseline = legacy_inputs(run.identity, "cuda")
    train = run.reader.open("train")
    plans = sealed_training_plans(train, LIMITS)
    rows, evidence = calibrate(run.reader, train, run.codec, run.suite, lock, baseline, plans,
                               run.config["calibration"]["videos"], LIMITS)
    run.verify_integrity()
    write_rows(OUTPUT / "calibration", rows, **evidence)
    write_json(CANDIDATES, {
        "protocol_sha256": file_hash(PROTOCOL), "calibration_sha256": file_hash(OUTPUT / "calibration.json"),
        "locked_at": timestamp(), "validation_pixels_read_before_lock": run.reader.counts["validation"],
        "test_images_read": run.reader.counts["test"], "shortlist": evidence["shortlist"], "arms": evidence["arms"],
    })
    print(f"Train shortlist frozen: {evidence['shortlist']}", flush=True)
    return frozen_candidates()


def validation_input(run):
    frozen_candidates()
    run.reader.validation_locked = True
    catalog = run.reader.open("validation")
    return catalog, validation_videos(run.config, catalog)


def validation_samples(run, catalog, counts, baseline):
    for video, count in counts.items():
        clip = run.reader.read(catalog, video, 0, count)
        for sample, elapsed in shared_frames(run.codec, clip, baseline):
            yield video, "validation", sample, elapsed
        print(f"Validation {video}, {count} frames/arm", flush=True)


def measure_arms(run, layouts, names):
    catalog, counts = validation_input(run)
    arms = {name: layouts[name] for name in names}
    samples = validation_samples(run, catalog, counts, layouts["A1"])
    histogram = Histograms()
    rows = list(evaluate_samples(run.codec, samples, arms, run.suite, histogram))
    return rows, histogram.record(), counts


def load_replay():
    record = json_at(REPLAY)
    if record["candidates_sha256"] != file_hash(CANDIDATES) or not record["replay"]["passed"]:
        raise RuntimeError("sealed historical A0/A1 replay must pass on the current lock")
    return record


def run_replay(run):
    if REPLAY.is_file() and json_at(REPLAY)["replay"]["passed"]:
        return load_replay()
    layouts = load_layouts(frozen_candidates(), "cuda")
    rows, histograms, counts = measure_arms(run, layouts, ARMS[:2])
    comparison = check_replay(rows, LEGACY_VALIDATION)
    write_rows(REPLAY.with_suffix(""), rows, replay=comparison, summaries=summarize_arms(rows),
               histograms=histograms, cohort=counts, candidates_sha256=file_hash(CANDIDATES))
    for name, result in comparison["comparisons"].items():
        print(f"{name} historical replay: {result}", flush=True)
    return load_replay()


def require_same_frames(rows, names, cohort):
    expected = {(name, video, index) for name in names for video, count in cohort.items() for index in range(count)}
    actual = {(row["arm"], *frame_key(row)[:2]) for row in rows}
    classes = {}
    for row in rows:
        key = (row["video"], row["frame"])
        previous = classes.setdefault(key, row["frame_type"])
        if previous != row["frame_type"]:
            raise RuntimeError("frame type differs across paired arms")
    if actual != expected or len(actual) != len(rows):
        raise RuntimeError("every arm requires all unique frames of the locked cohort")


def load_phase1():
    record = json_at(PHASE1)
    if record["candidates_sha256"] != file_hash(CANDIDATES) or record["validation_sha256"] != file_hash(VALIDATION):
        raise RuntimeError("Phase 1 gate disagrees with its Validation evidence")
    load_replay()
    lock = frozen_candidates()
    validation = json_at(VALIDATION)
    if validation["candidates_sha256"] != file_hash(CANDIDATES):
        raise RuntimeError("Validation used a different candidate lock")
    scores = {arm: summary["all"] for arm, summary in validation["summaries"].items()}
    expected = release_dense(lock["shortlist"], scores, LIMITS)
    if any(record[key] != value for key, value in expected.items()):
        raise RuntimeError("Phase 1 gate outcome disagrees with its locked scoring rule")
    return record


def run_validation(run):
    if PHASE1.is_file():
        return load_phase1()
    replay = load_replay()
    lock = frozen_candidates()
    layouts = load_layouts(lock, "cuda")
    rows, histograms, counts = measure_arms(run, layouts, ARMS[2:])
    rows = replay["rows"] + rows
    require_same_frames(rows, ARMS, counts)
    summaries = summarize_arms(rows)
    run.verify_integrity()
    write_rows(VALIDATION.with_suffix(""), rows, summaries=summaries, cohort=counts,
               histograms=replay["histograms"] + histograms, candidates_sha256=file_hash(CANDIDATES))
    decision = release_dense(lock["shortlist"], {arm: summary["all"] for arm, summary in summaries.items()}, LIMITS)
    write_json(PHASE1, {**decision, "candidates_sha256": file_hash(CANDIDATES),
                        "validation_sha256": file_hash(VALIDATION), "finished": timestamp()})
    print(f"Phase 1 release: {decision}", flush=True)
    return load_phase1()


def execute_stage(stage, run):
    if stage in ("all", "calibrate"):
        run_calibration(run)
    if stage in ("all", "replay"):
        run_replay(run)
    if stage in ("all", "validate"):
        run_validation(run)
    if stage in ("all", "train"):
        from .training import run_training

        run_training(run)
    if stage in ("all", "sparse"):
        from .sparse import run_sparse

        run_sparse(run)


def audit_path(stage):
    parent = OUTPUT / "audits"
    return parent / f"{stage}_{len(list(parent.glob(f'{stage}_*.json'))) + 1:02}.json"


def publish_report(audit, path):
    from .report import generate

    try:
        generate()
    except Exception as exc:
        write_json(path, audit | {"exit_code": 1, "report_error": f"{type(exc).__name__}: {exc}"})
        raise


def main(stage):
    with experiment_lock():
        if stage == "report":
            from .report import generate

            generate()
            return
        audit = {"stage": stage, "started": timestamp(), "command": sys.orig_argv,
                 "torch": str(torch.__version__), "cuda": torch.version.cuda}
        path = audit_path(stage)
        run = None
        try:
            audit["gpu"] = gpu_snapshot()
            assets = protected_digest()
            create_record(INTEGRITY, assets)
            run = start_experiment(assets)
            execute_stage(stage, run)
            audit["integrity"] = run.verify_integrity()
        except BaseException as exc:
            audit.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            audit["exit_code"] = 0
        finally:
            if run is not None:
                audit["images_read"] = run.reader.counts
                audit["denied_pixel_accesses"] = run.reader.denied
            audit["finished"] = timestamp()
            write_json(path, audit)
        if stage == "all":
            publish_report(audit, path)
