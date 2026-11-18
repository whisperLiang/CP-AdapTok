import pytest


def test_interventions_preserve_real_prefix_and_only_change_incremental_codes():
    from experiments.adaptok_hr.fine_detail_v2.evaluate import suffix_variants

    ids = tuple(range(12))
    donor = tuple(range(100, 112))
    shuffled, replaced = suffix_variants(ids, donor, 4, 18)
    assert shuffled[:4] == replaced[:4] == ids[:4]
    assert sorted(shuffled[4:]) == list(ids[4:]) and shuffled != ids
    assert replaced[4:] == donor[4:]
    assert suffix_variants(ids, donor, 4, 18) == (shuffled, replaced)
    assert suffix_variants(ids, None, 4, 18)[1] is None


def test_information_needs_both_content_controls_and_within_frame_variation():
    from experiments.adaptok_hr.fine_detail_v2.decisions import information

    def rows(deltas, diverse=True):
        return [dict(content_id=s, within_frame_used=(5 if diverse else 1), shuffle_mse_increase=a,
                     cross_mse_increase=b) for s, (a, b) in zip(("a", "b", "c"), deltas)]
    assert information(rows([(1e-5, 1e-5), (2e-5, 1e-5), (0, 0)]))["pass"]
    assert not information(rows([(1e-5, 0)] * 3))["pass"]
    assert not information(rows([(1e-5, 1e-5)] * 3, False))["pass"]
    assert not information(rows([(1e-5, 1e-5), (0, 0), (0, 0)]))["pass"]
    assert not information(rows([(1e-9, 1e-9)] * 3))["pass"]
