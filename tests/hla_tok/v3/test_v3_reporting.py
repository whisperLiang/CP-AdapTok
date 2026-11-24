from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v3 import report
from scripts.hla_tok.v3.protocol import PROTOCOL, read_config


def test_report_counts_authorized_extension_in_the_training_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUTPUT", tmp_path)
    monkeypatch.setattr(report, "state_of", lambda phase, seed: tmp_path / f"decisions/{phase}_seed{seed}.json")
    run = tmp_path / "continuous/seed1/B0"
    write_json(run / "training.json", {"optimizer_updates": 20500})
    write_json(run / "latest.json", {"step": 20500})
    write_json(tmp_path / "decisions/continuous_seed1_extension.json", {"authorized": True, "updates_per_arm": 4000})
    [row] = report.progress(read_config(PROTOCOL))
    assert row["planned_updates"] == 24000
    assert row["checkpoint_updates"] == 20500


def test_quantization_table_uses_real_fixed13_same_weights_and_continuous_control():
    common = {"global_psnr": 29.0, "lpips": 0.15, "dists": 0.07, "frames": 174, "matched_174": True}
    results = {
        "status": "PASS",
        "observed": common | {"global_psnr": 27.0, "lpips": 0.18, "dists": 0.08,
                              "mode": "fixed13", "actual_bpp": 0.10231481481481482},
        "immediate_fixed13_minus_bypass": {"global_psnr": -1, "lpips": 0.02, "dists": 0.004},
        "continuous_training_trajectory": {"global_psnr": -1, "lpips": 0.01, "dists": 0.006},
    }
    rows = report.quantization_gap({"C_seed1": results, "C_seed2": {"status": "BLOCKED"}})
    assert [row["metric"] for row in rows] == ["global_psnr", "lpips", "dists"]
    first = rows[0]
    assert first["C0_continuous"] == 29
    assert first["C1_bypass"] == 28
    assert first["C1_quantized"] == 27
    assert first["lookup_quality_gain"] == -1
    assert rows[1]["lookup_quality_gain"] == -0.02
    assert rows[2]["training_trajectory_quality_gain"] == -0.006


def test_frame_gains_keep_the_identity_of_regressing_frames():
    from hla_tok import BUDGETS

    def measured(video, frame, budget, psnr):
        return {"video": video, "frame": str(frame), "budget": str(budget), "mode": "fixed13",
                "psnr": str(psnr), "lpips": "0.1", "dists": "0.05"}

    frames = [measured("v", index, budget, 25 + (-1 if index == 1 else 1) * rank)
              for rank, budget in enumerate(BUDGETS) for index in range(2)]
    results = report.frame_marginal_gains(frames)
    assert len(results) == 7 * 2
    regressions = [row for row in results if row["psnr_gain_db"] < 0]
    assert len(regressions) == 7
    assert all(row["frame"] == 1 for row in regressions)
    assert regressions[0]["lower_budget"] == BUDGETS[0]
    assert regressions[0]["upper_budget"] == BUDGETS[1]


def test_report_reuses_observed_native_metrics_without_assigning_a_continuous_bitrate(tmp_path, monkeypatch):
    from hla_tok.evaluation.io import write_csv
    from scripts.hla_tok.v2.protocol import csv_at, json_at

    root = tmp_path / "v3"
    v2 = tmp_path / "frozen"
    output = root / "results/hla_tok/v3"
    old_results = root / "results/hla_tok/v2"
    write_json(v2 / "results/hla_tok/v2/acceptance.json", {})
    write_json(old_results / "acceptance.json", {"D2": {"paired_comparisons": []}})
    write_csv(old_results / "diag0/oracles/metrics.csv", [
        {"arm": "O0", "frames": 9, "global_psnr": 20, "lpips": 0.3, "dists": 0.15},
    ])
    write_csv(old_results / "benchmark/seed1/R2/timings.csv", [{"fps": 1}])
    write_csv(root / "results/hla_tok/baselines/metrics.csv", [{"frames": 174, "global_psnr": ""}])
    write_csv(root / "results/hla_tok/phase1/fixed/validation_metrics.csv", [
        {"frames": 174, "global_psnr": 22, "lpips": 0.2, "dists": 0.09, "actual_bpp": 0.1},
    ])
    for name, path in {"ROOT": root, "OUTPUT": output, "V2": old_results, "CHECKPOINTS": root / "checkpoints"}.items():
        monkeypatch.setattr(report, name, path)
    monkeypatch.setattr(report, "state_of", lambda phase, seed: output / f"decisions/{phase}_seed{seed}.json")

    dense = {"global_psnr": 31, "lpips": 0.11, "dists": 0.07, "frames": 174, "matched_174": True,
             "mode": "dense", "budget": None}
    write_json(output / "decisions/dense_seed1.json", {
        "status": "PASS", "selected": {"arm": "A0", "variant": "D0", "path": "/saved.pt", "sha256": "fixture"},
        "observed": {"A0": dense, "A1": dense | {"global_psnr": 31.05}},
    })
    write_json(output / "live_report/formal_state_audit.json", {
        "dense": {"A0": {"modules": {"encoder": "encoder hash", "rgb_decoder": "decoder hash"}}},
    })
    timing = {"precision": "bf16", "fps": 230}
    write_csv(output / "dense/seed1/A0/performance/timings.csv", [timing])
    write_json(output / "dense/seed1/A0/full/step_12000/metrics.json", [dense])

    b0 = output / "continuous/seed1/B0"
    observed = dense | {"mode": "continuous", "budget": 16320}
    write_json(output / "decisions/continuous_seed1.json", {"status": "PASS", "observed": observed})
    write_json(b0 / "training.json", {"optimizer_updates": 20000, "seen_pixels": 2112})
    write_json(b0 / "latest.json", {"step": 20000})
    measured = b0 / "full/step_20000"
    write_json(measured / "metrics.json", [observed])
    write_json(measured / "pixel_reads.json", {"test": 0, "validation": 174})
    frame = {key: 0.01 for key in report.FRAME_METRICS} | {
        "video": "v", "frame": 1, "budget": 16320, "mode": "continuous",
    }
    write_csv(measured / "per_frame.csv", [frame])

    destination = output / "snapshot"
    result = report.report(read_config(PROTOCOL), v2, destination)
    assert result["phase_a_selection"]["selected_encoder_sha256"] == "encoder hash"
    assert result["pixel_reads"]["test_pixel_reads"] == 0
    assert result["stages"]["B_seed1"]["status"] == "PASS"
    assert result["stages"]["C_seed1"]["status"] == "BLOCKED"
    assert json_at(destination / "acceptance.json")["status"] == "PARTIAL PASS"
    assert set(row["mode"] for row in csv_at(destination / "rate_distortion.csv")) == {"fixed13"}
    [by_video] = csv_at(destination / "reconstruction_metrics_by_video.csv")
    assert by_video["video"] == "v" and by_video["mode"] == "continuous"
