import json
from pathlib import Path

import torch

from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import validate_fast
from adaptok_hr.runtime import environment, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model


def main():
    torch.set_num_threads(4)
    seed_everything(20261117)
    root = Path("results/adaptok_hr")
    original = json.loads((root / "joint/validation_001000.json").read_text())["budgets"]
    model, _ = load_model("checkpoints/adaptok_hr/joint/step_001000.pt")
    current = validate_fast(model, load_index("data/adaptok_hr/index.json"),
                            ["DAVIS2017:blackswan", "DAVIS2017:car-roundabout", "DAVIS2017:dog"], frames=8)
    differences = {k: abs(row["mse"] - original[k]["mse"]) for k, row in current["budgets"].items()}
    write_json(root / "numeric_mode_validation.json", {
        "environment": environment(), "reference": "original frozen step1000 short validation before strict-GPU patch",
        "scope": "same 3x8 native pixels and frozen weights; quality audit, not a throughput benchmark",
        "absolute_mse_differences": differences, "maximum_absolute_mse_difference": max(differences.values()),
        "budgets": current["budgets"],
    })
    print(differences)


if __name__ == "__main__":
    main()
