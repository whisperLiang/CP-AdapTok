"""A joint regression gate tied to the actual receiver and native test inputs."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import ROOT, file_hash, write_json
from aip_vtok.sparse_rfsq.protocol import TEST_GATE as OLD_TEST_GATE
from aip_vtok.sparse_rfsq.receiver import RecoveryNet

from .models import ARMS, create_model
from .phase0 import require_phase0
from .protocol import OUTPUT, PROVENANCE

TESTS = OUTPUT / "phase1_tests.json"
ARCHITECTURES = OUTPUT / "phase1_architectures.json"
JUNIT = OUTPUT / "phase1_junit.xml"
NATIVE_AUDIT = OUTPUT / "phase1_test_inputs.json"


def architecture_counts(w1=None):
    models = {"E1": RecoveryNet() if w1 is None else w1} | {arm: create_model(arm) for arm in ARMS}
    return {"E0": 0} | {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}


def architecture_record():
    counts = architecture_counts()
    deviation = abs(counts["E3"] - counts["E4"]) / counts["E4"]
    if deviation > 0.05:
        raise RuntimeError(f"R-Wide and SLD exceed the parameter matching target: {deviation:.1%}")
    return {"parameters": counts, "r48g_gaussian_input_added_parameters": counts["E2"] - counts["E1"],
            "rwide_sld_parameter_relative_difference": deviation, "widths": {"E2": 48, "E3": 88, "E4": 64},
            "depthwise_blocks": {"E2": 4, "E3": 4, "E4": {"stem": 1, "full": 2, "coarse": 3, "fusion": 1}},
            "coarse_shape": [34, 60], "full_shape": [68, 120], "budget_embedding_channels": 8,
            "prediction_only_channels": 104, "step_zero": "zero residual heads, A2/SP-1 Gaussian B2",
            "precision": {"inference": "FP32", "training_rgb_decoder": "BF16 autocast"}}


def source_hashes():
    paths = [ROOT / "tests/aip_vtok/test_sparse_decoder.py",
             *(path for path in (ROOT / "aip_vtok/sparse_decoder").glob("*.py")
               if path.stem in {"evaluation", "models", "phase0", "phase1", "protocol"})]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def test_results():
    cases = ET.parse(JUNIT).findall(".//testcase")
    current = [case for case in cases if case.get("classname", "").split(".")[-1] == "test_sparse_decoder"]
    checks = {int(token[1:]) for case in current for token in case.get("name", "").split("_")
              if len(token) == 3 and token[0] == "t" and token[1:].isdigit()}
    failure = sum(any(node.tag in {"failure", "error", "skipped"} for node in case) for case in cases)
    return {"legacy": len(cases) - len(current), "new": len(current), "failed_or_skipped": failure,
            "contracts": sorted(checks)}


def require_phase1():
    phase0 = require_phase0()
    record = json_at(TESTS)
    if (record["exit_code"] != 0 or record["source_sha256"] != source_hashes()
            or record["provenance_sha256"] != file_hash(PROVENANCE) or record["phase0_outputs"] != phase0["outputs"]
            or record["junit_sha256"] != file_hash(JUNIT) or record["test_inputs_sha256"] != file_hash(NATIVE_AUDIT)
            or record["architectures_sha256"] != file_hash(ARCHITECTURES)):
        raise RuntimeError("Phase 1 results are missing, stale, or unverified")
    counts = test_results()
    if (record["counts"] != counts or counts["legacy"] != json_at(OLD_TEST_GATE)["counts"]["total"]
            or counts["failed_or_skipped"] or counts["contracts"] != list(range(1, 16))):
        raise RuntimeError(f"Pilot needs a passing joint regression and native contracts T1–T15: {counts}")
    inputs = json_at(NATIVE_AUDIT)
    if (not inputs["passed"] or inputs["images_read"] != {"train": 33, "validation": 0, "test": 0}
            or inputs["denied_reads"]):
        raise RuntimeError("Phase 1 tests require audited Train-only native inputs")
    return record


def run_phase1():
    phase0 = require_phase0()
    write_json(ARCHITECTURES, architecture_record())
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "cp_adaptok/tests", "tests/aip_vtok",
               f"--junitxml={JUNIT.relative_to(ROOT)}", "--cov=aip_vtok.sparse_decoder", "--cov-branch", "--cov-report="]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "SPARSE_DECODER_TEST_AUDIT": "1",
           "COVERAGE_FILE": str(OUTPUT / ".coverage.phase1")}
    for path in (JUNIT, NATIVE_AUDIT):
        path.unlink(missing_ok=True)
    use_source = source_hashes()
    with (OUTPUT / "phase1_pytest.log").open("w") as log:
        process = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    record = {"exit_code": process.returncode, "counts": test_results() if JUNIT.is_file() else None,
              "command": command, "source_sha256": use_source, "provenance_sha256": file_hash(PROVENANCE),
              "phase0_outputs": phase0["outputs"], "junit_sha256": file_hash(JUNIT) if JUNIT.is_file() else None,
              "test_inputs_sha256": file_hash(NATIVE_AUDIT) if NATIVE_AUDIT.is_file() else None,
              "architectures_sha256": file_hash(ARCHITECTURES)}
    write_json(TESTS, record)
    print(f"Phase 1 · pytest exit={process.returncode} · {record['counts']}", flush=True)
    require_phase1()
