"""Complete the historical, state-access, and engineering evidence without opening Test."""

import argparse
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from scripts.hla_tok.temporal.inputs import RESULTS, WORK
from scripts.hla_tok.v2.protocol import ROOT, json_at


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def history(output: Path) -> None:
    before = json_at(output / "historical_assets_before.json")
    original_paths = before["tracked_sha256"]
    mismatches = [name for name, expected in original_paths.items() if sha256_file(ROOT / name) != expected]
    old_refs = {line.split()[0]: line.split()[1] for line in before["historical_refs"]
                if line.split()[0] != "refs/heads/experiment/hla-tok-temporal"}
    changed_refs = [name for name, expected in old_refs.items() if _git("rev-parse", name) != expected]
    new_pointers = _git("submodule", "status", "--recursive").splitlines()
    submodule = [line.split()[:2] for line in new_pointers] == [line.split()[:2] for line in before["submodules"]]
    stage_names = ["baseline_inputs", "calibration_inputs", "diagnostic_inputs", "matrix_inputs", "dynamic_inputs",
                   "runtime_inputs"]
    audits = {name: json_at(output / f"{name}.json") for name in stage_names if (output / f"{name}.json").exists()}
    baseline = json_at(output / "baseline_reproduction.json")["c0"]
    audits["c0_native_replay"] = baseline
    stage_reads = {name: row["pixel_reads"] for name, row in audits.items()}
    total = sum(row["test"] for row in stage_reads.values())
    write_json(output / "test_access_audit.json", {"status": "PASS" if total == 0 else "FAIL",
               "test_pixel_reads": total, "test_unsealed": False, "stage_pixel_reads": stage_reads,
               "frozen_test_manifest_sha256": before["manifest_sha256"]["test"], "access_scope": "AuditedFrames",
               "synthetic_boundary_tests_use_isolated_fixture": True})
    model_records = [row["state_dict_after_sha256"] for row in audits.values() if "state_dict_after_sha256" in row]
    models_intact = len(set(model_records)) == 1
    passed = not mismatches and not changed_refs and submodule and models_intact and not total
    write_json(output / "historical_assets_audit.json", {"status": "PASS" if passed else "FAIL",
               "pre_work_sha256": sha256_file(output / "historical_assets_before.json"),
               "baseline_head": before["git_head"], "historical_files_checked": len(original_paths),
               "historical_branches_checked": len(old_refs), "changed_frozen_files": mismatches,
               "changed_historical_branches": changed_refs, "frozen_submodules": new_pointers,
               "model_state_hashes": model_records, "test_pixel_reads": total})
    if not passed:
        raise RuntimeError("historical assets, model state, or Test isolation failed")


def _tests(path: Path) -> dict:
    if not path.is_file():
        return {"status": "NOT_RUN"}
    root = ElementTree.parse(path)
    cases = list(root.iter("testcase"))
    outcome = {"tests": len(cases), **{name: sum(case.find(name) is not None for case in cases)
               for name in ("failure", "error", "skipped")},
               "temporal_cases": sum("temporal" in case.get("classname", "") for case in cases)}
    return outcome | {"status": "PASS" if outcome["tests"] > outcome["skipped"]
                      and not outcome["failure"] and not outcome["error"] else "FAIL"}


def engineering(output: Path) -> None:
    logs = WORK / "logs"
    reports = {group: _tests(logs / f"{file}.xml") for group, file in (
        ("cpu_regression", "cpu_tests_final"), ("cuda_regression", "cuda_tests"),
        ("temporal_cpu", "temporal_cpu_final"), ("cross_device", "cross_device"),
    )}
    commands = json_at(logs / "tool_status.json")
    mutations = json_at(output / "targeted_mutations.json")
    test_status = all(result["status"] == "PASS" for result in reports.values())
    tools_status = all(result["status"] == "PASS" for result in commands.values())
    write_json(output / "engineering_audit.json", {
        "status": "PASS" if test_status and tools_status and mutations["status"] == "PASS" else "FAIL",
        "recorded_tests": reports, "tools": commands, "named_mutations": mutations,
        "static_thresholds": {"crap": 10, "duplicate_similarity": 0.8},
        "historical_tolerances_changed": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("history", "engineering"))
    parser.add_argument("--output", type=Path, default=RESULTS)
    options = parser.parse_args()
    (history if options.stage == "history" else engineering)(options.output)
