

def test_interventions_preserve_real_prefix_and_only_change_incremental_codes():
    from experiments.adaptok_hr.fine_detail_v2.evaluate import suffix_variants

    ids = tuple(range(12))
    donor = tuple(range(100, 112))
    shuffled, replaced = suffix_variants(ids, donor, 4, 18)
    assert shuffled[:4] == replaced[:4] == ids[:4]
    assert sorted(shuffled[4:]) == list(ids[4:]) and shuffled != ids
    assert replaced[4:] == donor[4:]
    assert suffix_variants(ids, donor, 4, 18) == (shuffled, replaced)
    assert suffix_variants(ids, None, 4, 18)[1] is None


def test_information_needs_both_content_controls_and_within_frame_variation():
    from experiments.adaptok_hr.fine_detail_v2.decisions import information

    def rows(deltas, diverse=True):
        return [dict(content_id=s, within_frame_used=(5 if diverse else 1), shuffle_mse_increase=a,
                     cross_mse_increase=b) for s, (a, b) in zip(("a", "b", "c"), deltas)]
    assert information(rows([(1e-5, 1e-5), (2e-5, 1e-5), (0, 0)]))["pass"]
    assert not information(rows([(1e-5, 0)] * 3))["pass"]
    assert not information(rows([(1e-5, 1e-5)] * 3, False))["pass"]
    assert not information(rows([(1e-5, 1e-5), (0, 0), (0, 0)]))["pass"]
    assert not information(rows([(1e-9, 1e-9)] * 3))["pass"]


def test_quality_checks_absolute_controls_low_budget_and_late_frames():
    import copy

    from experiments.adaptok_hr.fine_detail_v2.decisions import quality

    base = dict(psnr_global=18., lpips=.77, dists=.64)
    loops = {str(k): {**base, "bytes": 3470 * 174, "frames": 174, "late": dict(base)}
             for k in (256, 512, 1024, 2048)}
    ref = {"closed_loop": loops}
    candidate = copy.deepcopy(ref)
    high = candidate["closed_loop"]["2048"]
    high.update(psnr_global=18.12, lpips=.76, dists=.63)
    candidate["same_state_gain_db"] = .11
    assert quality(candidate, ref, ref)["pass"]
    candidate["closed_loop"]["256"]["lpips"] += .006
    assert not quality(candidate, ref, ref)["checks"]["protected"]
    candidate["closed_loop"]["256"]["lpips"] -= .006
    high["late"]["psnr_global"] -= .06
    assert not quality(candidate, ref, ref)["checks"]["protected"]
    high["late"]["psnr_global"] += .06
    best = copy.deepcopy(ref)
    best["closed_loop"]["2048"]["dists"] = .62
    assert not quality(candidate, ref, best)["checks"]["absolute_quality"]
    high["bytes"] -= 1
    assert not quality(candidate, ref, ref)["checks"]["rate"]


def test_runner_recovers_if_final_checkpoint_precedes_completion(tmp_path, monkeypatch):
    import json

    from experiments.adaptok_hr.fine_detail_v2 import run

    monkeypatch.setattr(run, "ROOT", tmp_path / "results")
    monkeypatch.setattr(run, "CHECKPOINTS", tmp_path / "checkpoints")
    target = run.checkpoint("S1_R00", 300)
    target.parent.mkdir(parents=True)
    target.touch()
    target.with_suffix(".json").write_text("{}")
    calls = []
    monkeypatch.setattr(run, "execute", lambda command, log: calls.append(command))
    run.train(1, "R00", 300)
    assert len(calls) == 1 and calls[0][-2:] == ["--resume", target]
    output = run.ROOT / "S1_R00"
    output.mkdir(exist_ok=True)
    (output / "completion.json").write_text(json.dumps({"complete": True, "completed_steps": 300}))
    run.train(1, "R00", 300)
    assert len(calls) == 1


def test_benchmark_verifier_recomputes_rate_and_both_endpoint_timings():
    import copy
    import json
    from pathlib import Path

    import pytest

    from experiments.adaptok_hr.fine_detail_v2.verify import verify_timing

    saved = json.loads(Path("experiments/adaptok_hr/fine_recovery/evidence/A1/benchmark_001000.json").read_text())
    measured = saved["budgets"]["2048"]
    assert verify_timing(measured, 2048)["frames"] == 64
    wrong = copy.deepcopy(measured)
    wrong["frames"][8]["joint_ms"] = wrong["frames"][8]["sender_ms"]
    with pytest.raises(ValueError, match="latency"):
        verify_timing(wrong, 2048)
    wrong = copy.deepcopy(measured)
    wrong["frames"][32]["frame_type"] = "P"
    with pytest.raises(ValueError, match="GOP"):
        verify_timing(wrong, 2048)
    wrong = copy.deepcopy(measured)
    wrong["timing"]["joint"]["fps"] *= 2
    with pytest.raises(ValueError, match="aggregate"):
        verify_timing(wrong, 2048)
