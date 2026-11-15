"""Lifecycle boundaries, immutable artifacts, failed-update accounting and reporting."""

import copy
import csv
import json
from types import SimpleNamespace

import pytest

from cp_adaptok.large_scale import enforce_architecture, memory_fraction, restore_contract, scale_conditions
from cp_adaptok.large_scale_analysis import compare_reports, same_rate_points
from cp_adaptok.large_scale_tracking import TrainingObserver, append_csv, reconcile_training_logs, snapshot
from cp_adaptok.runtime import MissingPrerequisites
from cp_adaptok.training_state import stable_hash


def test_yaml_integer_budget_keys_survive_json_checkpoint_roundtrip():
    config = {"weights": {128: 1.0, 256: 1.0, 1024: 1.0, 2048: 1.0}, "grid": (17, 30)}
    assert stable_hash(config) == stable_hash(json.loads(json.dumps(config)))


def test_resume_at_update_target_finishes_missing_validation_even_with_an_older_best():
    from cp_adaptok.large_scale import final_validation_needed

    training = {"full_validate_every": 2000}
    state = {"best_by_tier": {"quick": {"step": 1000}}, "validations": [{"step": 1000, "tier": "quick"}]}
    assert final_validation_needed(state, 2000, training)
    state["validations"].append({"step": 2000, "tier": "quick"})
    assert final_validation_needed(state, 2000, training)
    state["validations"].append({"step": 2000, "tier": "full"})
    assert not final_validation_needed(state, 2000, training)
    assert final_validation_needed(state, 8000, training)


def test_recorded_configuration_reopens_but_changed_learning_rate_is_rejected(tmp_path):
    from cp_adaptok.large_scale import reuse_phase_configuration

    original = {"training": {"lr": 0.001}, "weights": {128: 1, 2048: 1}, "experiment": {}}
    stored = copy.deepcopy(original)
    path = tmp_path / "config.json"
    reuse_phase_configuration(stored, path, 0.8)
    path.write_text(json.dumps(stored))
    same = copy.deepcopy(original)
    reuse_phase_configuration(same, path, 0.8)
    assert same["experiment"]["config_hash"] == stored["experiment"]["config_hash"]
    with pytest.raises(MissingPrerequisites, match="headroom"):
        reuse_phase_configuration(copy.deepcopy(original), path, 0.6)
    changed = copy.deepcopy(original)
    changed["training"]["lr"] = 0.002
    with pytest.raises(ValueError, match="different phase"):
        reuse_phase_configuration(changed, path, 0.8)


def test_final_selection_is_sealed_before_test_and_cannot_change(tmp_path):
    from cp_adaptok.large_scale_analysis import evaluate_experiment, seal_selection

    manifest = {"run_id": "test", "manifest_hashes": {"test": "held-out"}}
    directory = tmp_path / "test"
    directory.mkdir()
    selected = {"B0": {"checkpoint": "fixed", "sha256": "aaa"}}
    seal_selection(tmp_path, manifest, selected)
    seal_selection(tmp_path, manifest, selected)
    assert manifest["final_test_frozen"]
    with pytest.raises(ValueError, match="already sealed"):
        seal_selection(tmp_path, manifest, {"B0": {"checkpoint": "different", "sha256": "bbb"}})
    with pytest.raises(MissingPrerequisites, match="S1-S4"):
        evaluate_experiment({"phases": {}}, {})


def test_plot_curves_uses_real_recorded_values_and_delta(tmp_path):
    from cp_adaptok.large_scale_analysis import delta_psnr_groups, plot_curves

    train = {"run_id": "test", "phase": "s1", "branch": "B0", "step": "1", "loss": "0.1"}
    append_csv(tmp_path / "training_metrics.csv", list(train), train)
    validation = [{"run_id": "test", "phase": "s1", "branch": "B0", "tier": "quick", "step": "1",
                   "tokens": str(k), "psnr": str(p)} for k, p in ((128, 20), (2048, 21))]
    for row in validation:
        append_csv(tmp_path / "validation_metrics.csv", list(row), row)
    assert delta_psnr_groups(validation)["s1/B0 quick"] == [(1, 1)]
    paths = plot_curves(tmp_path, "test")
    assert len(paths) == 3
    assert all(__import__("pathlib").Path(p).stat().st_size > 1000 for p in paths)


def test_resource_limit_uses_current_free_memory_without_claiming_exclusivity():
    settings = {"gpu_min_free_mib": 40000, "gpu_reserve_mib": 8000}
    assert memory_fraction({"total_mib": 80000, "free_mib": 79000}, settings) == 0.8
    assert memory_fraction({"total_mib": 80000, "free_mib": 60000}, settings) == 0.65
    with pytest.raises(MissingPrerequisites):
        memory_fraction({"total_mib": 80000, "free_mib": 20000}, settings)


def test_architecture_and_dataset_gate_forbid_silent_substitution():
    with pytest.raises(ValueError, match="n_max"):
        enforce_architecture({"model": {"n_max": 1024}})
    with pytest.raises(ValueError, match="token_aggregation"):
        enforce_architecture({"model": {"token_aggregation": "shared_progressive_v2"}})
    rows = [{"content_id": "train:1", "resolution": [1280, 720], "original_frames": 100,
             "native_1080p_eligible": False, "supported_crops": [[512, 512], [720, 1280]]}]
    settings = {"minimum_train_videos": 100, "minimum_train_frames": 10000, "minimum_native1080_videos": 8}
    failures = scale_conditions({"train": rows}, settings)
    assert len(failures) == 3
    assert "100 actual training frames" in failures[1]


def test_inference_artifact_and_incomplete_rng_cannot_resume():
    config = {"experiment": dict(phase="s1", branch="B0", config_hash="a", manifest_hashes={}, structure_hash="b")}
    artifact = {"config": config, "inference_only": True}
    with pytest.raises(ValueError, match="inference"):
        restore_contract(config, artifact)
    artifact.pop("inference_only")
    with pytest.raises(ValueError, match="random streams"):
        restore_contract(config, artifact)


def test_snapshot_is_immutable_and_idempotent_when_source_replaced(tmp_path):
    source, best = tmp_path / "latest.pt", tmp_path / "best.pt"
    source.write_bytes(b"first")
    snapshot(source, best)
    snapshot(source, best)
    assert not (tmp_path / "best.tmp").exists()
    new = tmp_path / "new"
    new.write_bytes(b"second")
    new.replace(source)
    assert best.read_bytes() == b"first"


def test_resume_preserves_failed_work_without_double_counting(tmp_path):
    identity = {"run_id": "run", "phase": "s1", "branch": "B0"}
    output, root = tmp_path / "phase", tmp_path / "all"
    output.mkdir()
    root.mkdir()
    records = [{"step": step, "frames": 8} for step in (1, 2, 3)]
    for name in ("train.jsonl", "samples.jsonl"):
        (output / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    for row in records:
        append_csv(root / "training_metrics.csv", [*identity, "step", "frames"], {**identity, **row})
    append_csv(root / "training_metrics.csv", [*identity, "step", "frames"],
               {**identity, "branch": "B1", "step": 4, "frames": 8})
    config = {"experiment": identity, "paths": {"output": str(output)}, "large_scale": {"output_root": str(root)}}
    reconcile_training_logs(config, 2)
    reconcile_training_logs(config, 2)
    assert len((output / "train.jsonl").read_text().splitlines()) == 2
    assert len((output / "train.jsonl.uncheckpointed.jsonl").read_text().splitlines()) == 1
    with (root / "training_metrics.csv").open() as handle:
        remaining = list(csv.DictReader(handle))
    assert [(r["branch"], r["step"]) for r in remaining] == [("B0", "1"), ("B0", "2"), ("B1", "4")]


def test_validation_keeps_best_absolute_quality_checkpoint(tmp_path, monkeypatch):
    config = {"experiment": {"run_id": "run", "phase": "s1", "branch": "B0", "target_updates": 2,
                              "config_hash": "a", "structure_hash": "b", "manifest_hashes": {}},
              "paths": {"output": str(tmp_path / "out"), "save_checkpoint": str(tmp_path / "latest.pt")},
              "large_scale": {"output_root": str(tmp_path), "selection_rule": "fixed"},
              "training": {"validate_every": 1, "full_validate_every": 100}}
    (tmp_path / "out").mkdir()
    manifest = {"upstreams": []}
    observer = TrainingObserver(config, {}, manifest)
    score = [0.01]

    def validation(*args, **kwargs):
        rows = [{"tokens": 128, "mse": score[0], "psnr": 20}, {"tokens": 2048, "mse": 0.009, "psnr": 21}]
        return {"budget_summary": rows, "budget_summary_by_frame_type": {"I": rows, "P": rows},
                "quality_drop_ratio": 0}

    monkeypatch.setattr("cp_adaptok.large_scale_tracking.validate_checkpoint", validation)
    source = tmp_path / "latest.pt"
    source.write_bytes(b"good absolute base")
    model = SimpleNamespace(vq=SimpleNamespace(strategy="deterministic_nearest"))
    state = {}
    assert observer.validate(model, 1, source, state)
    replacement = tmp_path / "new.pt"
    replacement.write_bytes(b"larger prefix gain but worse base")
    replacement.replace(source)
    score[0] = 0.05
    observer.validate(model, 2, source, state)
    assert (tmp_path / "best_validation.pt").read_bytes() == b"good absolute base"
    assert state["best_by_tier"]["quick"]["step"] == 1


def test_paired_comparison_uses_same_frames_and_never_extrapolates_rates():
    def report(delta):
        rows = [{"tokens": k, "bpp": k / 10000, "psnr": 20 + delta + i} for i, k in enumerate((128, 256, 512, 1024, 2048))]
        return {"records": [{"source": {"content_id": name}, "frame_id": 0, "prefixes": copy.deepcopy(rows)}
                            for name in ("a", "b", "c")]}

    a, b = report(0), report(1)
    comparison = compare_reports(a, b)
    assert all(row["mean_video_delta_psnr"] == 1 for row in comparison["sequence_bootstrap"])
    assert all(row["delta_psnr"] == pytest.approx(1) for row in comparison["same_bpp_interpolation"])
    assert comparison["single_training_seed"] and not comparison["equal_flops_claim"]
    b["records"].pop()
    with pytest.raises(ValueError, match="identical"):
        compare_reports(a, b)
    assert same_rate_points(("a", 1), [{"bpp": 1, "psnr": 1}], [{"bpp": 2, "psnr": 2}]) == []
