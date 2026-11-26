"""Reduce measured test and tool logs into the V3 acceptance ledger."""

import argparse
import re
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import json_at


def pytest_counts(path):
    log = path.read_text()
    result = {label: int(count) for count, label in re.findall(r"(\d+) (passed|failed|skipped|deselected|error)", log)}
    if "passed" not in result:
        raise ValueError(f"{path.name} has no pytest success count")
    return result


def read_crap(log):
    worst = []
    for row in log.splitlines():
        captured = re.search(r"\s(\d+)\s+(\d+\.\d+)%\s+(\d+\.\d+)$", row)
        if captured:
            worst.append(float(captured[3]))
    if not worst:
        raise ValueError("Crapper score table was not produced")
    return max(worst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    out, logs = args.output, args.logs
    mutation = {label: json_at(out / f"mutations-{label}/summary.json")["counts"]
                for label in ("cpu", "cuda", "v3")}
    selection = json_at(out.parent / "runtime_trial/selection.json")
    contracts = json_at(out.parent / selection["contracts"])
    report = {"cpu": pytest_counts(logs / "cpu.log"), "cuda": pytest_counts(logs / "cuda.log"),
              "directed_mutations": mutation, "actual_stale_cuda_injection":
                  sum(row["stale_cuda_detected"] for row in contracts["deployment"]["changes"]),
              "lookahead_injection": contracts["future"]["future_leak_mutation"],
              "detail_only_mutation": "NOT_RUN",
              "mypy_new_source": (logs / "mypy-v3.log").read_text().strip(),
              "mypy_inherited_scope": "4 inherited errors in V2 decoder/records with this checker",
              "ruff": (logs / "ruff.log").read_text().strip(),
              "crapper": {"max": read_crap((logs / "crapper.log").read_text()), "scope": "hla_tok/uf_v3"},
              "dryer": {"result": (logs / "dryer.log").read_text().splitlines()[-1], "similarity_threshold": 0.8},
              "mutator": {"killed": (logs / "mutator.log").read_text().count("KILLED    "), "scope": "selected sites"},
              "workspace_pack_cpu": pytest_counts(logs.parent / "pack-reuse/cpu.log")}
    write_json(out / "summary.json", report)


if __name__ == "__main__":
    main()
