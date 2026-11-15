"""Content evidence excludes decoder position vectors and preserves reference IDs."""

import torch

from cp_adaptok.diagnostics_prefix import intervene_ids, token_group_statistics


def test_group_content_variance_excludes_position_embedding():
    ids = torch.ones(1, 32, dtype=torch.long)
    content = torch.ones(1, 32, 12)
    positions = torch.randn_like(content)
    report = token_group_statistics(ids, content, positions, 8, 32)
    assert all(row["content_variance"] == 0 for row in report["groups"])
    assert all(row["unique_ids"] == 1 for row in report["groups"])
    assert all(row["positioned_variance"] > 0 for row in report["groups"])
    assert report["between_group_js_divergence"][0][1] == 0


def test_interventions_preserve_base_and_do_not_mutate_source():
    ids = torch.arange(32)[None]
    original = ids.clone()
    donor = ids.flip(-1)
    for mode in ("same_frame", "other_frame", "shuffle", "constant"):
        changed = intervene_ids(ids, donor, 8, mode, torch.Generator().manual_seed(5))
        assert torch.equal(changed[:, :8], original[:, :8])
        assert not torch.equal(changed[:, 8:], original[:, 8:])
        assert (changed >= 0).all() and (changed < 32).all()
    assert torch.equal(ids, original)
