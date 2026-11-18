"""Behavioral mutations supplement the repository's retained codec mutations."""

import json

from adaptok_hr.runtime import write_json
from experiments.adaptok_hr.quality import MUTATIONS, run_mutation

from .run import ROOT

NEW = [
    ("pair_mean", "training/train.py", "loss = (loss + extra) * .5", "loss = loss + extra", "test_unified.py::test_paired_gradient_is_mean_of_independent_budgets_and_only_main_commits"),
    ("crop_restore", "training/checkpoint.py", 'trainer.losses.crop_rng.set_state(state["crop_rng"].cpu())', "pass  # discard crop history", "test_unified.py::test_unified_crop_and_budget_rng_resume_exactly"),
    ("crop_coordinates", "training/losses.py", "target[..., y:y+h, x:x+w] * 2 - 1", "target[..., :h, :w] * 2 - 1", "test_unified.py::test_crop_views_use_identical_native_coordinates_and_independent_rng"),
    ("fusion_gradient", "models/encoder.py", "outputs.append(refine(value))", "outputs.append(feature)", "test_unified.py::test_cross_scale_shapes_gradients_positions_and_shared_initialization"),
    ("paired_probability", "training/train.py", 'paired < self.settings.get("paired_probability", .5)', "paired < .5", "test_unified.py::test_paired_draws_match_across_frequencies_and_are_balanced"),
]


if __name__ == "__main__":
    rows = []
    for case in [*MUTATIONS, *NEW]:
        row = run_mutation(case)
        rows.append(row)
        print(json.dumps({"mutation": row["mutation"], "killed": row["killed"]}), flush=True)
    write_json(ROOT / "mutations.json", {"all_killed": all(r["killed"] for r in rows), "results": rows})
