import copy
import json

import pytest
import torch

from experiments.adaptok_hr.unified.evaluate import color_metrics, grouped_summary


def test_color_metric_identity_and_gray_chroma_variance():
    frame = torch.rand(1, 3, 1080, 1920)
    same = color_metrics(frame, frame)
    assert same["cb_mae"] == same["cr_mae"] == 0
    assert same["cb_variance"] == same["cb_target_variance"]
    gray = frame.mean(1, keepdim=True).expand_as(frame)
    lost = color_metrics(frame, gray)
    assert lost["cb_mae"] > .01 and lost["cr_mae"] > .01
    assert lost["cb_variance"] < 1e-12 and lost["cr_variance"] < 1e-12
    rows = [{**same, "mse": .01, "psnr": 20., "lpips": .1, "dists": .2, "pixels": 1080*1920,
             "tokens": 2048, "bytes": 3470, "header_bytes": 142, "payload_bytes": 3328,
             "frame": i, "frame_type": "I" if i % 32 == 0 else "P", "content_id": "video"} for i in range(64)]
    value = grouped_summary(rows)
    assert value["color_difference_variance_retention"] == pytest.approx(1.)
    assert value["I"]["frames"] == 2 and value["P"]["frames"] == 62
    assert value["late_GOP"]["frames"] == 16
    assert value["actual_bpp"] == pytest.approx(3470*8/(1080*1920))


def fixture_quality(lpips=.5, psnr=20.):
    return {"closed_loop": {str(k): {"lpips": lpips, "dists": .3, "psnr_global": psnr}
                            for k in (256, 512, 1024, 2048)},
            "same_state_gain_db": .1, "same_state_lpips_gain": .01,
            "full_fine": {variant: {"mse_increase": 1e-6} for variant in ("shuffle", "cross")}}


def test_pilot_selection_requires_both_seeds_and_guards_every_budget():
    from experiments.adaptok_hr.unified.decision import pilot_decision

    measurements = {str(s): {arm: fixture_quality(.5 if arm == "U0" else .48)
                             for arm in ("U0", "U1", "U2", "U3")} for s in (1, 2)}
    decision = pilot_decision(measurements)
    assert decision["unified_improves_both_seeds"]
    assert decision["preferred_action"] == "optimize_training"
    failed = copy.deepcopy(measurements)
    for arm in ("U1", "U2", "U3"):
        failed["2"][arm] = fixture_quality(.51)
    decision = pilot_decision(failed)
    assert not decision["unified_improves_both_seeds"]
    assert decision["preferred_action"] == "analyze_cross_scale"
    for arm in ("U1", "U2", "U3"):
        measurements["2"][arm]["closed_loop"]["256"]["psnr_global"] = 19.8
    assert not pilot_decision(measurements)["preferred_training_gate_pass"]


def test_scratch_gate_stops_growth_for_one_bad_budget_or_one_seed():
    from experiments.adaptok_hr.unified.decision import scratch_quality_gate

    data = {str(s): {arm: fixture_quality(.5 if arm == "N0" else .48)
                    for arm in ("N0", "N1", "N2")} for s in (1, 2)}
    assert scratch_quality_gate(data)["advance"]
    for arm in ("N1", "N2"):
        data["2"][arm]["closed_loop"]["256"]["lpips"] = .51
    assert not scratch_quality_gate(data)["advance"]


def test_scratch_audit_rejects_optimizer_history(small_model, tmp_path, monkeypatch):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.tests.test_unified import unified_settings
    from adaptok_hr.training.checkpoint import save_training
    from adaptok_hr.training.train import JointTrainer
    from experiments.adaptok_hr.unified import run, verify

    monkeypatch.setattr(run, "CHECKPOINTS", tmp_path / "checkpoints")
    config = {"model": small_model.config.to_dict(), "training": {**unified_settings(), "model_seed": 7}}
    trainer = JointTrainer(small_model, config["training"])
    sampler = ClipSampler([{"original_frames": 8}], 1)
    path = run.checkpoint("S1_N0", 0)
    save_training(path, trainer, sampler, config, "native-test", 0, 0)
    state = {"config": config}
    assert verify.scratch_initialization("S1_N0", state)
    trainer.update(torch.rand(2, 3, 128, 256), 1)
    # A checkpoint relabeled as step zero must not pass the scratch audit.
    save_training(path, trainer, sampler, config, "native-test", 0, 0)
    with pytest.raises(ValueError, match="history"):
        verify.scratch_initialization("S1_N0", state)


def test_300_screen_records_quality_but_1000_controls_long_training(monkeypatch):
    from experiments.adaptok_hr.unified import decision

    data = {str(s): {arm: fixture_quality(.5 if arm == "N0" else .7, 20. if arm == "N0" else 19.)
                    for arm in ("N0", "N1", "N2")} for s in (1, 2)}
    monkeypatch.setattr(decision, "stability_screen", lambda step, arms=None: ({f"S{s}_{a}": {"finite_gradients": True} for s in (1, 2) for a in ("N0", "N1", "N2")}, True))
    assert decision.scratch_decision(data, 300)["advance"]
    assert not decision.scratch_decision(data, 1000)["advance"]
    monkeypatch.setattr(decision, "stability_screen", lambda step, arms=None: ({f"S{s}_{a}": {"finite_gradients": False} for s in (1, 2) for a in ("N0", "N1", "N2")}, False))
    assert not decision.scratch_decision(data, 300)["advance"]


def test_stable_passing_architecture_is_not_blocked_by_rejected_comparator(monkeypatch):
    from experiments.adaptok_hr.unified import decision

    data = {str(s): {arm: fixture_quality({"N0": .5, "N1": .48, "N2": .7}[arm])
                    for arm in ("N0", "N1", "N2")} for s in (1, 2)}
    stable = {f"S{s}_{arm}": {"finite_gradients": True, "final_no_collapse": arm != "N2"}
              for s in (1, 2) for arm in ("N0", "N1", "N2")}
    monkeypatch.setattr(decision, "stability_screen", lambda step, arms=None: (stable, False))
    result = decision.scratch_decision(data, 1000)
    assert result["advance"]
    assert result["continuation_arms"] == ["N0", "N1"]


def test_runner_reuses_earlier_completed_checkpoint_without_downgrading_marker(tmp_path, monkeypatch):
    from adaptok_hr.runtime import write_json
    from experiments.adaptok_hr.unified import run

    monkeypatch.setattr(run, "ROOT", tmp_path / "results")
    monkeypatch.setattr(run, "CHECKPOINTS", tmp_path / "checkpoints")
    target = run.checkpoint("S1_N0", 300)
    target.parent.mkdir(parents=True)
    target.touch()  # Completed-stage reuse must not invoke the trainer or load this sentinel.
    completion = run.ROOT / "S1_N0/completion.json"
    saved = {"completed_steps": 1000, "complete": True}
    write_json(completion, saved)
    calls = []
    monkeypatch.setattr(run, "execute", lambda *args: calls.append(args))
    run.train("S1_N0", {"training": {}}, 300)
    assert not calls
    assert json.loads(completion.read_text()) == saved


def test_native_archive_uses_only_architectures_measured_at_latest_gate(tmp_path, monkeypatch):
    from adaptok_hr.runtime import write_json
    from experiments.adaptok_hr.unified import archive

    root, evidence = tmp_path / "results", tmp_path / "evidence"
    monkeypatch.setattr(archive, "ROOT", root)
    monkeypatch.setattr(archive, "EVIDENCE", evidence)
    write_json(root / "decision_A.json", {"winner": "U1"})
    write_json(root / "decision_B_3000.json", {"advance": False})
    for seed in (1, 2):
        for arm, step, lpips in (("U1", 1600, .5), ("N0", 3000, .4),
                                 ("N1", 3000, .3), ("N2", 1000, .1)):
            directory = root / f"S{seed}_{arm}" / f"validation_{step}_64"
            write_json(directory / "summary.json", {"closed_loop": {"2048": {"lpips": lpips}}})
            samples = directory / "samples"
            samples.mkdir()
            (samples / "car-roundabout_0031_K2048.png").write_bytes(f"{seed}/{arm}/{step}".encode())
    archive.selected_native_images()
    assert {p.name for p in (evidence / "native_samples").iterdir()} == {"S1_U1", "S2_U1", "S1_N1", "S2_N1"}
    assert (evidence / "native_samples/S2_N1/car-roundabout_0031_K2048.png").read_bytes() == b"2/N1/3000"
