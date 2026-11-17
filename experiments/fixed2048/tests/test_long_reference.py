from experiments.fixed2048.long_reference import summarize


def test_long_i_only_has_no_p_window_statistics():
    rows = [
        {
            "video": "validation:a",
            "frame": i,
            "frame_type": "I",
            "reference": "i_only",
            "path": "R1",
            "group": "i_only/R1",
            "mse": 0.01,
            "psnr": 20,
            "pixels": 1,
        }
        for i in range(48)
    ]
    summary = summarize(rows)["i_only/R1"]
    assert summary["P"] is None
    assert summary["early_P"] is None and summary["late_P"] is None
    assert summary["global_mse_psnr"] == 20
