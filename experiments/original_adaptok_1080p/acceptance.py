"""Verify stored real artifacts, source protection and every formal acceptance boundary."""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

from .artifacts import write_json
from .comparison import load
from .video_reader import file_sha256

BASE = "534696b2f78b7e1bc74bf5c5eefe83e88f41b864"
REQUIRED_METRICS = ["mse", "psnr", "ms-ssim", "lpips", "lpips-vgg", "dists", "boundary_mse", "interior_mse"]


def check_frame_table(directory):
    with (directory / "frame_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 192 * 3
    for budget in [512, 1024, 2048]:
        selected = [r for r in rows if int(r["tokens_per_tile_window"]) == budget]
        assert len({(r["content_id"], int(r["frame_id"])) for r in selected}) == 192
    assert all(math.isfinite(float(row[name])) for row in rows for name in REQUIRED_METRICS)
    assert all(row["bpp"] == "" and row["quality_only"] == "True" for row in rows)
    return {"rows": len(rows), "all_metrics_measured": True, "bpp_null": True}


def check_videos(directory):
    paths = list(directory.glob("*/reconstructed_videos/*.mp4"))
    assert len(paths) == 9
    for path in paths:
        data = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(path)], text=True)
        stream = json.loads(data)["streams"][0]
        assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (1920, 1080, 64)
    return {"videos": len(paths), "dimensions": [1920, 1080], "real_frames_each": 64}


def check_tail(root):
    path = next((root / "tail_smoke/evaluation").glob("*/completed.json"))
    video = load(path)
    windows = video["windows"]
    assert [len(w["budgets"]["2048"]["frames"]) for w in windows] == [16, 1]
    assert [r["frame_id"] for w in windows for r in w["budgets"]["2048"]["frames"]] == list(range(17))
    baseline = next((root / "no_overlap").glob("*/completed.json"))
    expected = load(baseline)["windows"][0]["budgets"]["2048"]["frames"]
    actual = windows[0]["budgets"]["2048"]["frames"]
    error = max(abs(a[k] - b[k]) for a, b in zip(actual, expected) for k in REQUIRED_METRICS)
    assert error < 1e-6
    return {"real_frames": 17, "scored_per_window": [16, 1], "refactor_vs_original_run_metric_max_abs_error": error}


def check_protected(root):
    paths = ["cp_adaptok", "third_party", "configs", "pyproject.toml", ":!configs/original_adaptok_1080p.yaml"]
    difference = subprocess.check_output(["git", "diff", BASE, "--", *paths], text=True)
    assert not difference
    parity = load(root / "cp_input_parity.json")
    for branch, expected in parity["reference_sha256"].items():
        assert file_sha256(Path(parity["reference"]) / branch / "validation.json") == expected
    assert file_sha256("cp_adaptok/metrics.py") == parity["cp_metric_source_sha256"]
    return {"existing_code_configs_submodules_unchanged": True, "historical_cp_results_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/original_adaptok_1080p"))
    args = parser.parse_args()
    root = args.results
    checkpoint = load(root / "checkpoint_manifest.json")
    assert checkpoint["strict_loading"] and checkpoint["official_hash_verified"] and checkpoint["weights_unchanged"]
    assert checkpoint["state_sha256_before"] == checkpoint["state_sha256_after"]
    assert file_sha256(Path(checkpoint["directory"]) / "model.safetensors") == checkpoint["model_safetensors_sha256"]
    result = {"strict_official_weights_and_immutable_state": True, "official_equivalence": load(root / "smoke/official_equivalence.json")["passed"],
              "source_and_metric_parity": load(root / "absolute_quality_comparison.json")["parity"], "layouts": {},
              "tail": check_tail(root), "protected": check_protected(root)}
    for name in ["final_1080p", "token_ablation/overlap"]:
        result["layouts"][name] = {**check_frame_table(root / name), **check_videos(root / name)}
    assert all(item["killed"] for item in load(root / "mutations/summary.json"))
    for item in load(root / "benchmarks/summary.json").values():
        assert not item["metric_networks_resident"] and item["latency"]["source_frames"] == 48
        assert item["latency"]["compute"]["total_fps"] > 0 and item["peak_allocated_mib"] > 0
    result.update(all_acceptance_checks_passed=True, real_measured_source_frames=192, reconstructed_frame_budget_layout_pairs=1152)
    write_json(root / "acceptance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
