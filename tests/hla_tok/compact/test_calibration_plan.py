from scripts.hla_tok.compact.inputs import train_plan


def source(index, *, resized=False):
    return {"role": "train", "resolution": [1920, 1080], "resized": resized,
            "frame_hashes": [{"sha256": f"sha{index}{frame}"} for frame in range(5)]}


def test_calibration_is_deterministic_source_disjoint_and_native():
    catalog = {str(i): source(i, resized=i == 6) for i in range(7)}
    config = {"seed": 4, "fit_frames": 12, "holdout_frames": 8, "holdout_sources": 2}
    plan = train_plan(catalog, config)
    assert plan == train_plan(dict(reversed(list(catalog.items()))), config)
    assert len(plan["fit"]) == 12
    assert len(plan["holdout"]) == 8
    fit = {row["video"] for row in plan["fit"]}
    holdout = {row["video"] for row in plan["holdout"]}
    assert len(fit) == 4 and len(holdout) == 2
    assert not fit & holdout
    assert "6" not in fit | holdout
    for row in plan["fit"] + plan["holdout"]:
        assert row["source_sha256"] == catalog[row["video"]]["frame_hashes"][row["frame"]]["sha256"]


def test_small_cohort_uses_every_legal_frame():
    config = {"seed": 4, "fit_frames": 96, "holdout_frames": 32, "holdout_sources": 8}
    plan = train_plan({str(i): source(i) for i in range(3)}, config)
    assert len(plan["fit"]) + len(plan["holdout"]) == 15
    assert plan["fit"] and plan["holdout"]
