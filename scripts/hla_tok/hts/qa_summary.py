"""Collect the executed quality checks for the native Phase 0 evidence."""

import argparse
import re
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import json_at


def pytest_status(path: Path) -> dict:
    if not path.is_file():
        return {"verdict": "NOT_RUN"}
    raw = path.read_text()
    counts = {}
    for key in ("passed", "failed", "skipped", "deselected"):
        matches = re.findall(rf"(\d+) {key}\b", raw)
        counts[key] = int(matches[-1]) if matches else 0
    return counts | {"verdict": "PASS" if counts["passed"] and not counts["failed"] else "FAIL"}


def executed(logs: Path, name: str) -> dict:
    report = pytest_status(logs / f"{name}.log")
    code = logs / f"{name}.exit"
    if code.is_file():
        report["exit_code"] = int(code.read_text())
    return report


def contract_status(native: Path) -> dict:
    results = dict.fromkeys((f"T{i}" for i in range(1, 31)), "NOT_RUN")
    verdict = json_at(native / "native_correctness.json")["verdict"]
    for index in (1, 2, 3, 4, 5, 6, 21, 24):
        results[f"T{index}"] = verdict
    data_closed = json_at(native / "pixel_reads.json")
    results["T26"] = "PASS" if not (data_closed["validation"] or data_closed["test"]) else "FAIL"
    if (native / "byte_accounting.json").is_file() and verdict == "PASS":
        results.update(T18="PARTIAL_PASS", T22="PARTIAL_PASS", T30="PARTIAL_PASS")
    return results | {"native_baseline": verdict,
                      "scope": "Actual Phase 0 official codec and data-boundary contracts."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    output = args.results / "qa"
    data = {
        "cpu_regression": executed(args.logs, "cpu_regression_final"),
        "cuda_regression": executed(args.logs, "cuda_regression"),
        "cuda_hts": executed(args.logs, "cuda_hts_verified"),
        "cpu_mutations": json_at(output / "mutations-final/summary.json"),
        "cuda_mutations": json_at(output / "mutations-gpu-verified/summary.json"),
        "contracts": contract_status(args.results / "native_audit"),
    }
    write_json(output / "summary.json", data)


if __name__ == "__main__":
    main()
