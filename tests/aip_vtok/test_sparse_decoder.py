import pytest

from aip_vtok.sparse_decoder.evaluation import CLOSED, FIXED, validate_rows
from aip_vtok.sparse_decoder.phase0 import contrasts


def test_factorial_effects_are_oriented_and_include_history():
    def point(value):
        return {"psnr_global_mse": value, "psnr_frame_mean": 2 * value,
                "lpips": -value, "dists": value / 10}

    fixed = dict(zip(("Q1", "Q2", "Q3", "Q4"), map(point, (10, 7, 11, 9))))
    closed = dict(zip(("Q1", "Q2", "Q3", "Q4"), map(point, (10, 6, 12, 8))))
    actual = contrasts(fixed, closed)
    assert actual[FIXED]["scan_a1"]["psnr_global_mse"] == -3
    assert actual[FIXED]["quant_old"]["lpips"] == -1
    assert actual[FIXED]["quant_sp1"]["dists"] == pytest.approx(0.2)
    assert actual[FIXED]["interaction"]["psnr_global_mse"] == 1
    assert actual[CLOSED]["interaction"]["psnr_frame_mean"] == 0
    assert actual["closed_minus_fixed"]["Q4_minus_Q1"]["psnr_global_mse"] == -1


def test_inventory_rejects_dropped_duplicate_and_unverified_rows():
    row = {"video": "clip", "frame": 0, "arm": "Q1", "history": FIXED, "budget": 510,
           "state_sync": True, "prefix_ids_match": True}
    expected = {("Q1", FIXED, 510)}
    validate_rows([row], {"clip": 1}, expected)
    for rows in ([], [row, row], [row | {"prefix_ids_match": False}], [row | {"frame": 1}]):
        with pytest.raises(RuntimeError):
            validate_rows(rows, {"clip": 1}, expected)
