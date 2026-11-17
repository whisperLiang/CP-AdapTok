"""Count which levels own VQ supervision; report evidence without asserting training causality."""

import json
from pathlib import Path

from adaptok_hr.config import ModelConfig
from adaptok_hr.models.tokenizer import PositionTable
from adaptok_hr.runtime import write_json

from .diagnose import LEVELS


def fractions(counts, budget):
    offset, result = 0, {}
    for name, capacity in zip(LEVELS, counts):
        result[name] = max(0, min(capacity, budget - offset)) / budget
        offset += capacity
    return result


def main():
    root = Path("results/adaptok_hr_fine_recovery")
    counts = PositionTable(ModelConfig()).counts
    result = {"capacities": dict(zip(LEVELS, counts)), "all_candidate_loss_fraction": fractions(counts, sum(counts)),
              "actual_prefix_fractions": {str(k): fractions(counts, k) for k in (256, 512, 1024, 2048)}, "experiments": {}}
    for name in ("A0", "A1", "A2", "A3"):
        rows = [json.loads(line) for line in (root / name / "training.jsonl").read_text().splitlines()]
        sums = {level: 0. for level in LEVELS}
        observations = 0
        for row in rows:
            for frame in row["frame_metrics"]:
                budgets = [frame["main_budget"]]
                if frame["auxiliary_budget"] is not None:
                    budgets.append(frame["auxiliary_budget"])
                for budget in budgets:
                    for level, fraction in fractions(counts, budget).items():
                        sums[level] += fraction / len(budgets)
                observations += 1
        result["experiments"][name] = {"updates": len(rows), "frames": observations,
              "reconstruction_matched_loss_fraction": {level: value / observations for level, value in sums.items()},
              "observed_parameter_updates": [{"step": row["step"], "norms": row["module_update_norms"]}
                                                for row in rows if "module_update_norms" in row]}
    result["premise_to_reconsider_if_gate_fails"] = (
        "Changing VQ loss coverage or codebook ownership alone is sufficient to make the existing Fine encoder/decoder carry spatial detail. "
        "This is an unproven premise, not a measured conclusion. Do not assign all responsibility to the shared codebook if both controlled fixes fail. "
        "There are three representation actors (Base/Medium/Fine), one existing shared encoder hierarchy and one existing fused decoder. "
        "Without effective Fine, remove the assignment to longer training and stop; a future redesigned representation needs its own evidence and protocol.")
    write_json(root / "mechanism_audit.json", result)


if __name__ == "__main__":
    main()
