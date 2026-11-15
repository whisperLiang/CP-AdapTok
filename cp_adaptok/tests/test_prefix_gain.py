from cp_adaptok.evaluation import prefix_gain_summary


def frame(sequence, delta, frame_type="P"):
    return {"sequence": sequence, "frame_type": frame_type,
            "prefixes": [{"tokens": 8, "psnr": 20.0, "lpips": 0.5, "dists": 0.3},
                         {"tokens": 32, "psnr": 20.0 + delta, "lpips": 0.5 - delta / 100,
                          "dists": 0.3 - delta / 200}]}


def test_bootstrap_samples_sequences_and_is_invariant_to_frame_replication():
    records = [frame("a", 1.0), frame("b", -0.5), frame("c", 0.5)]
    a = prefix_gain_summary(records, bootstrap_samples=500)
    b = prefix_gain_summary(records * 20, bootstrap_samples=500)
    assert a[0]["psnr"]["sequence_bootstrap_ci95"] == b[0]["psnr"]["sequence_bootstrap_ci95"]
    assert a[0]["independent_sequences"] == 3
    assert a[0]["psnr"]["regression_fraction"] == 1 / 3
    assert a[0]["lpips"]["mean_improvement"] > 0


def test_single_sequence_cannot_establish_between_video_confidence():
    rows = prefix_gain_summary([frame("a", 1), frame("a", 2)])
    assert rows[0]["psnr"]["sequence_bootstrap_ci95"] is None
    assert rows[0]["independent_sequences"] == 1
