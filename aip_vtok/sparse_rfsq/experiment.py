"""Single-writer entry points for the S1 selection and conditional S2 pilot."""

import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch

from aip_vtok.c1.artifacts import frozen_inputs, load_frozen, timestamp
from aip_vtok.c1.metrics import metric_suite
from aip_vtok.c1_followup.io import ReadAudit, tensor_digest
from aip_vtok.diagnostics.diag0.io import command, gpu_snapshot, json_at
from aip_vtok.runtime import ROOT, file_hash, write_json
from aip_vtok.training.determinism import configure_determinism

from .calibration import Inputs, freeze_scan, load_inputs, load_scan
from .protocol import (
    BASE,
    BRANCH,
    OUTPUT,
    TEST_GATE,
    experiment_lock,
    freeze_protocol,
    protected_digest,
    source_digest,
    write_integrity,
)

BASE_TESTS = 306
TEST_XML = OUTPUT / "regression.xml"


@dataclass
class Experiment:
    config: dict
    codec: object
    suite: object
    reader: ReadAudit
    inputs: Inputs
    assets_before: dict
    frozen_before: dict

    def frozen_modules(self):
        return {name: getattr(self.codec, name) for name in ("backbone", "projector")}

    def tensor_hashes(self):
        return {name: tensor_digest(model) for name, model in self.frozen_modules().items()}

    def verify_integrity(self):
        after = self.tensor_hashes()
        assets = protected_digest()
        gradients = [f"{part}.{name}" for part, model in self.frozen_modules().items()
                     for name, param in model.named_parameters() if param.requires_grad or param.grad is not None]
        passed = assets == self.assets_before and after == self.frozen_before and not gradients
        passed = passed and self.reader.counts["test"] == 0 and self.reader.denied == 0
        record = {"passed": passed, "assets_before": self.assets_before, "assets_after": assets,
                  "tensors_before": self.frozen_before, "tensors_after": after, "frozen_gradients": gradients,
                  "test_images_read": self.reader.counts["test"], "denied_reads": self.reader.denied}
        if not passed:
            raise RuntimeError(f"protected assets, model parameters or data access changed: {record}")
        return record


def open_run(assets):
    configure_determinism()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, identity = frozen_inputs()
    freeze_protocol(config, identity)
    run = Experiment(config, load_frozen(config), metric_suite(), ReadAudit(config, manifest_hashes=identity["manifests"]),
                     load_inputs(config, identity), assets, {})
    run.frozen_before = run.tensor_hashes()
    return run


def test_counts():
    cases = ET.parse(TEST_XML).findall(".//testcase")
    new = sum(c.get("classname", "").split(".")[-1] == "test_sparse_rfsq" for c in cases)
    failed = sum(any(node.tag in {"failure", "error", "skipped"} for node in c) for c in cases)
    return {"total": len(cases), "regression": len(cases) - new, "new": new, "failed_or_skipped": failed}


def test_gate():
    record = json_at(TEST_GATE)
    counts = test_counts()
    if (record["exit_code"] != 0 or record["source_sha256"] != source_digest()
            or record["junit_sha256"] != file_hash(TEST_XML) or record["counts"] != counts
            or counts["regression"] != BASE_TESTS or not counts["new"] or counts["failed_or_skipped"]):
        raise RuntimeError("S2 requires a passing joint suite with all 306 legacy tests and new contracts")
    return record


def run_tests():
    started = timestamp()
    source = source_digest()
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "cp_adaptok/tests", "tests/aip_vtok",
            f"--junitxml={TEST_XML.relative_to(ROOT)}", "--cov=aip_vtok.sparse_rfsq", "--cov-branch", "--cov-report="]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "COVERAGE_FILE": str(OUTPUT / ".coverage.regression")}
    with (OUTPUT / "regression.log").open("w") as log:
        process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    counts = test_counts() if TEST_XML.is_file() else None
    write_json(TEST_GATE, {"started": started, "finished": timestamp(), "command": argv, "exit_code": process.returncode,
                           "counts": counts, "source_sha256": source,
                           "junit_sha256": file_hash(TEST_XML) if counts else None})
    print(f"joint regression: {counts}, exit={process.returncode}", flush=True)
    test_gate()


def execute_stage(stage, run):
    if stage == "calibrate":
        freeze_scan(run)
        return
    if stage in {"s1", "all"}:
        from .evaluation import run_s1

        run_s1(run, freeze_scan(run))
    if stage in {"s2", "all"}:
        from .training import run_pilot

        run_pilot(run, load_scan(), test_gate())


def audit_path(stage):
    directory = OUTPUT / "audits"
    return directory / f"{stage}_{len(list(directory.glob(stage + '_*.json'))) + 1:02}.json"


def idle_gpu(wait):
    announced = False
    while True:
        try:
            return gpu_snapshot()
        except RuntimeError as exc:
            if not wait or "competing PIDs ['" not in str(exc):
                raise
            if not announced:
                print("H100 busy. Waiting for the active GPU work to finish.", flush=True)
                announced = True
            time.sleep(2)


def preflight(audit, wait):
    branch = command("git", "branch", "--show-current")
    base = command("git", "rev-parse", BASE)
    if branch != BRANCH or command("git", "merge-base", "HEAD", BASE) != base:
        raise RuntimeError(f"experiment requires branch {BRANCH} based on {BASE}")
    audit["gpu"] = idle_gpu(wait)
    assets = protected_digest()
    write_integrity(assets)
    return assets


def main(stage, wait_for_gpu=False):
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
            assets = preflight(audit, wait_for_gpu)
            if stage in {"test", "all"}:
                run_tests()
                audit["test_gate_sha256"] = file_hash(TEST_GATE)
            if stage != "test":
                if stage == "all":
                    audit["gpu_after_regression"] = idle_gpu(wait_for_gpu)
                run = open_run(assets)
                execute_stage(stage, run)
            audit["integrity"] = run.verify_integrity() if run else {"passed": protected_digest() == assets}
            if not audit["integrity"]["passed"]:
                raise RuntimeError("protected input hashes changed")
            audit["exit_code"] = 0
        except BaseException as exc:
            audit.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if run is not None:
                audit["images_read"] = run.reader.counts
                audit["denied_pixel_accesses"] = run.reader.denied
            audit["finished"] = timestamp()
            write_json(path, audit)
        if stage == "all":
            from .report import generate

            generate()
