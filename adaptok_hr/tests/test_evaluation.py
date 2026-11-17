import math

import pytest


def test_psnr_global_and_frame_mean_stay_distinct():
    from adaptok_hr.evaluation.evaluate import aggregate, group_rows

    rows = [{"mse": mse, "psnr": -10 * math.log10(mse), "lpips": .2, "dists": .1,
             "bytes": 558, "payload_bytes": 416, "header_bytes": 142, "tokens": 256,
             "pixels": 1920 * 1080, "encode_ms": 10., "decode_ms": 8.,
             "frame_type": kind, "gop_position": pos, "motion": .02}
            for mse, kind, pos in ((.01, "I", 0), (.04, "P", 31))]
    result = aggregate(rows)
    assert result["psnr_global"] == pytest.approx(-10 * math.log10(.025))
    assert result["psnr_frame_mean"] == pytest.approx(16.98970004336)
    assert result["actual_bpp"] == pytest.approx(558 * 8 / (1920 * 1080))
    assert result["theoretical_code_bpp"] == pytest.approx(13 * 256 / (1920 * 1080))
    assert result["bytes"] == 1116
    assert group_rows(rows)["late_P"]["frames"] == 1


def test_same_numeric_k_is_not_used_as_equal_rate_baseline():
    from adaptok_hr.evaluation.report import baseline_table

    records = [{"model": "Official stride128", "budget_units": "2048/tile/native-window",
                "content_id": "all", "psnr_mean_db": 24., "lpips": .2, "dists": .1, "bpp": None},
               {"model": "B1", "budget_units": "2048/frame", "content_id": "all",
                "psnr_mean_db": 15., "lpips": .5, "dists": .6, "bpp": None}]
    table = baseline_table(records)
    assert "tile/native-window" in table and "2048/frame" in table
    assert "24.000" in table and "15.000" in table
