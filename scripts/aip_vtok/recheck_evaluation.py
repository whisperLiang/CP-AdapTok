"""Repeat identical validation into a new namespace after an abnormal session exit."""

import csv
import json

import aip_vtok.evaluation.formal as evaluation
from aip_vtok.runtime import resolve, write_json
from aip_vtok.training.formal import settings


def main():
    destination = resolve("results/aip_vtok/recheck/gate1_seed2")
    original_assets = evaluation.evaluation_assets

    def redirected_assets(config):
        assets = original_assets(config)
        return (destination, *assets[1:])

    evaluation.evaluation_assets = redirected_assets
    try:
        evaluation.evaluate(settings("configs/aip_vtok/gate1.yaml", 2))
    finally:
        evaluation.evaluation_assets = original_assets
    original = resolve("results/aip_vtok/gate1_seed2")
    with (original / "frame_metrics.csv").open() as handle:
        first = list(csv.DictReader(handle))
    with (destination / "frame_metrics.csv").open() as handle:
        second = list(csv.DictReader(handle))
    result = {
        "seed": 2,
        "rows": len(second),
        "all_csv_fields_equal": first == second,
        "evaluation_json_equal": json.loads((original / "evaluation.json").read_text())
        == json.loads((destination / "evaluation.json").read_text()),
        "primary_session_exit_code": 143,
        "primary_artifacts_completed_before_exit": len(first) == 1740,
        "scope": "same checkpoint, input, precision and algorithms; output directory redirected only",
    }
    write_json("results/aip_vtok/evaluation_recheck.json", result)
    print(result)
    if not result["all_csv_fields_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
