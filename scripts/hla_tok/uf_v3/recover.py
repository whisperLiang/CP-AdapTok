"""Record the cap failure and select the first joint checkpoint through predeclared Base gates."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import json_at


def only_candidate(path):
    rows = list(json_at(path)["candidates"].values())
    if len(rows) != 1:
        raise ValueError("expected a physical recheck of one candidate")
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", type=Path)
    args = parser.parse_args()
    diagnostics = args.out / "diagnostics"
    combined = json_at(diagnostics / "base_probe/summary.json")
    late = combined["candidates"]["r0-probe-seed1-step2064"]
    first = only_candidate(diagnostics / "base_probe_first/summary.json")
    confirmation = diagnostics / "base_probe_first_validation/summary.json"
    validation = only_candidate(confirmation) if confirmation.exists() else None
    candidates = args.checkpoints
    if late["status"] != "FAIL":
        raise ValueError("recovery is reserved for an observed late Base cap failure")
    passed = first["status"] == "PASS" and validation is not None and validation["status"] == "PASS"
    chosen = candidates / ("r0-probe-seed1-step2001.pt" if passed else "r0-P2-seed1-step2000.pt")
    late_gate = args.out / "pilot/probe/gate.json"
    write_json(late_gate, {"status": "FAIL", "reason": "BASE_BUDGET_EXCEEDED", "physical_base": late,
                           "strict_quality_and_runtime": "NOT_RUN", "seed2": "NOT_RUN"})
    write_json(diagnostics / "base_recovery.json", {
        "hard_cap": combined["cap"], "late_checkpoint": late, "first_joint_checkpoint": first,
        "validation_check": validation, "selected_research_candidate": chosen.name,
        "reason": "one direct checkpoint replay of the first native joint update with the original rate threshold",
        "formal_training": "NOT_RUN", "seed2": "NOT_RUN", "test_pixel_reads": 0,
    })
    (args.out / "training/candidate.txt").write_text(str(chosen) + "\n")
    print(chosen)


if __name__ == "__main__":
    main()
