from pathlib import Path

import pytest

from hla_tok import BUDGETS
from hla_tok.evaluation.io import write_json
from hla_tok.streaming.fixed13 import payload_ledger
from scripts.hla_tok.v2.protocol import json_at
from scripts.hla_tok.v3 import downstream
from scripts.hla_tok.v3.protocol import PROTOCOL, read_config


def _quality(budget, *, global_psnr=28.0, lpips=0.14, dists=0.07):
    frames = 174
    return {"budget": budget, "mode": "fixed13", "frames": frames, "matched_174": True,
            "global_psnr": global_psnr, "lpips": lpips, "dists": dists} | payload_ledger(
                ((budget * 13 + 7) // 8) * frames, budget * frames, frames, 1080, 1920)


@pytest.fixture
def fake_checkpoints(tmp_path, monkeypatch):
    source = tmp_path / "parent.pt"
    source.write_bytes(b"registered checkpoint")
    decisions = tmp_path / "decisions"

    def decision(stage, seed):
        return decisions / f"{stage}_seed{seed}.json"

    def model_from_checkpoint(path):
        assert path == source
        return object(), {"source": str(path)}

    for name, body in {
        "state_of": decision, "_approved_checkpoint": lambda decision: source,
        "_context": lambda *args: (None, None, None, None), "_digest": lambda model: "model digest",
        "model_from_checkpoint": model_from_checkpoint, "on_gpu": lambda model: model,
        "_selected": lambda run: {"step": 24000, "variant": run.trial.variant},
        "extension_active": lambda *args: False,
    }.items():
        monkeypatch.setattr(downstream, name, body)
    write_json(decision("continuous", 1), {"status": "PASS", "selected": {"variant": "D0"}})
    write_json(decision("quantized", 2), {"status": "PASS", "selected": {"variant": "D0"}})
    monkeypatch.setattr(downstream, "OUTPUT", tmp_path)
    return decision


@pytest.mark.parametrize(("allow_extension", "expected"), ((True, "PASS"), (False, "FAIL")))
def test_fixed_trials_repeat_both_arms_only_for_the_authorized_extension(
    fake_checkpoints, tmp_path, monkeypatch, allow_extension, expected,
):
    events = []

    def execute(run, *, extend):
        arm = run.trial.arm
        events.append((arm, extend))
        control = {"mode": "continuous", "global_psnr": 30, "lpips": 0.12, "dists": 0.06}
        if arm == "C0":
            return [control]
        terminal = _quality(16320, dists=0.08 if extend else 0.10)
        return [terminal, control | {"mode": "same_checkpoint_bypass", "global_psnr": 29}]

    def measure(run, budgets, precisions):
        assert (run.trial.arm, budgets, precisions) == ("C1", (16320,), ("bf16",))
        events.append("performance")
        return {"pipeline_fps": 30}

    monkeypatch.setattr(downstream, "execute", execute)
    monkeypatch.setattr(downstream, "_matching", lambda runs: events.append("match"))
    monkeypatch.setattr(downstream, "_calibrate_quantizer", lambda run: events.append("calibrate"))
    monkeypatch.setattr(downstream, "_performance_and_contract", measure)
    monkeypatch.setattr(downstream, "_extension", lambda *args: allow_extension)

    config = read_config(PROTOCOL)
    result = downstream.quantized_trials(config, tmp_path, 1)
    assert result["status"] == expected
    assert events[:4] == [("C0", False), "calibrate", ("C1", False), "match"]
    if allow_extension:
        assert events[4:] == [("C0", True), ("C1", True), "match", "performance"]
    else:
        assert len(events) == 4
    assert result["immediate_fixed13_minus_bypass"]["global_psnr"] == -1
    assert result["continuous_training_trajectory"]["global_psnr"] == -1
    assert json_at(fake_checkpoints("quantized", 1)) == result


def test_progressive_trials_bind_both_verdicts_to_all_eight_budget_timings(fake_checkpoints, tmp_path, monkeypatch):
    write_json(fake_checkpoints("quantized", 1), {"status": "PASS", "selected": {"variant": "D0"}})
    events = []

    def execute(run, *, extend):
        events.append((run.trial.seed, extend))
        return [_quality(budget, global_psnr=23 + i, lpips=0.22 - 0.015 * i, dists=0.109 - 0.01 * i)
                for i, budget in enumerate(BUDGETS)]

    def measure(run, budgets, precisions):
        assert budgets == BUDGETS
        assert precisions == ("bf16", "fp32")
        events.append(f"timing seed {run.trial.seed}")
        return {"pipeline_fps": 31 if run.trial.seed == 1 else 29}

    monkeypatch.setattr(downstream, "execute", execute)
    monkeypatch.setattr(downstream, "_fixed_trends", lambda run: [])
    monkeypatch.setattr(downstream, "_performance_and_contract", measure)
    monkeypatch.setattr(downstream, "_nonmonotonic", lambda run, step: events.append(f"frame seed {run.trial.seed}"))

    result = downstream.progressive_trials(read_config(PROTOCOL), Path(tmp_path))
    assert result["status"] == "FAIL"
    assert events == [(1, False), (2, False), "frame seed 1", "timing seed 1", "frame seed 2", "timing seed 2"]
    assert json_at(fake_checkpoints("progressive", 1))["status"] == "PASS"
    assert json_at(fake_checkpoints("progressive", 2))["status"] == "FAIL"


@pytest.mark.parametrize(("psnr", "status", "diagnoses"), ((31, "PASS", 0), (24, "BLOCKED", 1)))
def test_continuous_gate_only_schedules_location_diagnostics_after_quality_failure(
    fake_checkpoints, tmp_path, monkeypatch, psnr, status, diagnoses,
):
    from types import SimpleNamespace

    from torch import nn

    from scripts.hla_tok.v3 import training

    write_json(fake_checkpoints("dense", 1), {"status": "PASS", "selected": {"variant": "D0"}})
    monkeypatch.setattr(training, "OUTPUT", tmp_path)
    dense = SimpleNamespace(encoder=nn.Linear(2, 2))
    candidate = object()
    reader = SimpleNamespace(counts={"train": 0, "validation": 0, "test": 0})
    events = []
    observed = _quality(16320, global_psnr=psnr) | {"mode": "continuous"}

    def train(run, *, extend):
        assert not extend and run.model is candidate
        assert run.teacher is not dense.encoder
        assert not run.teacher.training
        assert all(not param.requires_grad for param in run.teacher.parameters())
        return [observed]

    monkeypatch.setattr(downstream, "_seed_r2", lambda *args: tmp_path / "parent.pt")
    monkeypatch.setattr(downstream, "model_from_checkpoint", lambda path: (dense, None))
    monkeypatch.setattr(downstream, "_make_continuous", lambda *args: candidate)
    monkeypatch.setattr(downstream, "_context", lambda *args: (reader, {}, None, None))
    monkeypatch.setattr(downstream, "calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(downstream, "execute", train)
    monkeypatch.setattr(downstream, "_extension", lambda *args: False)
    monkeypatch.setattr(downstream, "diagnose", lambda *args: events.append(args))

    verdict = downstream.continuous_trials(read_config(PROTOCOL), tmp_path, 1)
    assert verdict["status"] == status
    assert len(events) == diagnoses
    assert bool(verdict["selected"]) == (status == "PASS")
    assert json_at(fake_checkpoints("continuous", 1)) == verdict
