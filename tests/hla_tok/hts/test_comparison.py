from pytest import approx

from scripts.hla_tok.hts.native_comparison import compared_points, strict_fit


def point(rate, psnr, lpips):
    return {"actual_mbps": rate, "global_psnr": psnr, "lpips": lpips}


def test_equal_quality_comparison_requires_overlap_and_saves_physical_bytes():
    reference = [point(0.28, 25, 0.45), point(0.40, 28, 0.35)]
    candidate = [point(0.21, 24, 0.50), point(0.26, 25, 0.42), point(0.45, 29, 0.34)]
    pairs = compared_points(candidate, reference, "global_psnr", 0.486)
    assert len(pairs) == 1
    assert pairs[0]["reference_mbps"] == approx(0.28)
    assert pairs[0]["candidate_mbps"] == approx(0.26)
    assert pairs[0]["rate_saving_percent"] == approx(100 * (1 - 0.26 / 0.28))


def test_strict_fit_charges_all_bytes_against_valid_frame_time():
    rows = [
        {"unit_type": "I", "valid_frames": 1, "total_bytes": 2050},
        {"unit_type": "P", "valid_frames": 8, "total_bytes": 1800},
        {"unit_type": "P", "valid_frames": 1, "total_bytes": 1800},
    ]
    result = strict_fit(rows, 0.486, 30)
    assert result["affordable_units"] == 2
    assert result["affordable_valid_frames"] == 9
    assert result["units"] == 3
    assert result["valid_frames"] == 10
    assert result["i_units_over_budget"] == 1


def test_native_rates_charge_tail_bytes_using_source_duration():
    import torch

    from scripts.hla_tok.hts.audit_native import Source, aggregate
    from scripts.hla_tok.temporal.inputs import Clip

    source = Source(Clip("train", "example", 6, 2), 24, torch.empty(0))
    units = [
        {"unit_type": "I", "total_bytes": 12, "native_payload_bytes": 8, "header_bytes": 4,
         "encoding_ms": 3.0, "decoding_ms": 2.0, "valid_frames": 1, "padded_frames": 0, "lpips": 0.4},
        {"unit_type": "P", "total_bytes": 21, "native_payload_bytes": 18, "header_bytes": 3,
         "encoding_ms": 7.0, "decoding_ms": 3.0, "valid_frames": 1, "padded_frames": 7, "lpips": 0.3},
    ]
    frame_metrics = [{"mse": 0.001, "lpips": 0.4, "dists": 0.22, "psnr": 30.0},
                     {"mse": 0.01, "lpips": 0.3, "dists": 0.17, "psnr": 20.0}]
    result = aggregate([source], units, frame_metrics)
    assert result["actual_mbps"] == approx(33 * 8 * 30 / 2 / 1e6)
    assert result["actual_mbps_source_fps"] == approx(33 * 8 * 24 / 2 / 1e6)
    assert result["padded_frames"] == 7
    assert result["global_psnr"] == approx(22.5963731)
    assert result["i_peak_bytes"] == 12
    assert result["p_peak_bytes"] == 21


def test_incomplete_calibration_cannot_admit_any_training(tmp_path):
    from scripts.hla_tok.hts.phase0_gate import CONFIG, inspect
    from scripts.hla_tok.v2.protocol import json_at

    result = inspect(tmp_path, json_at(CONFIG))
    assert result["verdict"] == "BLOCKED"
    assert len(result["missing_native_cases"]) == 408
    assert "native_correctness.json" in result["missing_evidence"]
