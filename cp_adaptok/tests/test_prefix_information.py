"""Content evidence excludes decoder position vectors and preserves reference IDs."""

import pytest
import torch

from cp_adaptok.diagnostics_prefix import (
    attention_diagnostics,
    gradient_diagnostics,
    intervene_ids,
    token_group_statistics,
)


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


def test_constant_control_uses_the_same_code_across_different_frames():
    for first in (5, 7):
        ids = (torch.arange(32)[None] + first) % 32
        constant_suffix = intervene_ids(ids, ids.flip(-1), 8, "constant")
        assert torch.equal(constant_suffix[:, :8], ids[:, :8])
        assert torch.count_nonzero(constant_suffix[:, 8:]) == 0
        assert torch.count_nonzero(intervene_ids(ids, ids, 8, "all_constant")) == 0


def test_gradient_report_uses_reconstruction_graph_and_never_attributes_missing_tokens(codec):
    encoded = codec.encode_full(torch.rand(1, 3, 16, 24), codec.empty_state())
    report = gradient_diagnostics(codec, encoded, torch.rand(1, 3, 16, 24), [8, 16, 32])
    gradients = report["budgets"][0]
    assert gradients["decoder_embedding"][0]["l2"] > 0
    assert all(group["l2"] == 0 for group in gradients["decoder_embedding"][1:])
    assert all(group["l2"] == 0 for group in gradients["vq_input"][1:])
    assert len(report["conflicts"]) == 2
    components = attention_diagnostics(codec.decoder.projector, encoded.vq.embeddings.detach(), [8, 32], 8)
    assert abs(sum(components[-1]["group_attention_mass"]) - 1) < 1e-6


@pytest.mark.parametrize("progressive", [False, True])
def test_attention_diagnostics_never_write_parameters_or_invalidate_live_graph(progressive):
    from cp_adaptok.decoder.progressive_token_aggregator import SharedProgressiveTokenAggregator
    from cp_adaptok.decoder.token_grid_projector import TokenGridProjector

    cls = SharedProgressiveTokenAggregator if progressive else TokenGridProjector
    kwargs = {"packet_tokens": 8} if progressive else {}
    projector = cls(24, 4, 32, (2, 3), **kwargs)
    tokens = torch.randn(1, 32, 24, requires_grad=True)
    loss = projector(tokens).square().mean()
    versions = [p._version for p in projector.parameters()]
    report = attention_diagnostics(projector, tokens, [8, 32], 8)
    assert report[-1]["position_grid_rmse"] > 0
    assert [p._version for p in projector.parameters()] == versions
    assert torch.autograd.grad(loss, tokens)[0].norm() > 0
