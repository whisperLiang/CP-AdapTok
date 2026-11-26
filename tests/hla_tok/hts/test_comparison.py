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
