from aip_vtok.evaluation.gates import assess_gate0, distortion_summary


def test_psnr_global_mse_is_not_arithmetic_frame_psnr():
    summary = distortion_summary([{"mse": 0.001}, {"mse": 0.1}])
    assert abs(summary["psnr_frame_mean"] - 20) < 1e-8
    assert abs(summary["psnr_global_mse"] - 12.9670862188) < 1e-8


def test_unmeasured_or_failed_gate_never_releases_next_stage():
    limits = {"minimum_r0_psnr_db": 32, "maximum_ref_drop_db": 0.5, "maximum_projector_ms": 1.0}
    baseline = {
        "r0_native": {"frames": 64, "psnr_global_mse": 33.0},
        "r0_aligned": {"frames": 64, "psnr_global_mse": 32.8},
    }
    assert not assess_gate0(baseline, [], limits)["passed"]
    trial = {
        "summary": {"frames": 64, "psnr_global_mse": 32.0},
        "projector_latency": {"maximum_mean_ms": 0.9},
        "training_steps": 400,
    }
    assert not assess_gate0(baseline, [trial], limits)["passed"]
    trial["summary"]["psnr_global_mse"] = 32.6
    assert assess_gate0(baseline, [trial], limits)["passed"]
    trial["projector_latency"]["maximum_mean_ms"] = 1.0
    assert not assess_gate0(baseline, [trial], limits)["passed"]
    baseline["r0_aligned"]["frames"] = 63
    assert not assess_gate0(baseline, [trial], limits)["passed"]
