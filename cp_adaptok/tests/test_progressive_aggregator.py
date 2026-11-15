import pytest
import torch

from cp_adaptok.decoder.progressive_token_aggregator import SharedProgressiveTokenAggregator
from cp_adaptok.decoder.token_grid_projector import TokenGridProjector
from cp_adaptok.tokenizer.prefix_masking import prefix_mask


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_incremental_groups_match_every_dense_prefix_and_only_project_new_groups(precision):
    torch.manual_seed(17)
    projector = SharedProgressiveTokenAggregator(24, 4, 64, (3, 4), precision, 8).eval()
    tokens = torch.randn(2, 64, 24)
    cache = projector.new_cache(2)
    calls = []
    hook = projector.kv.register_forward_hook(lambda module, args, result: calls.append(args[0].shape[1]))
    with torch.no_grad():
        first = projector.append(cache, tokens[:, :16], 0)
        assert calls == [8, 8]
        calls.clear()
        last = projector.append(cache, tokens[:, 16:32], 16)
        assert calls == [8, 8]
    hook.remove()
    torch.testing.assert_close(first, projector(tokens[:, :16]), rtol=0, atol=0)
    torch.testing.assert_close(last, projector(tokens[:, :32]), rtol=0, atol=0)
    for start in range(32, 64, 8):
        cached = projector.append(cache, tokens[:, start:start + 8], start)
        torch.testing.assert_close(cached, projector(tokens[:, :start + 8]), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(projector.decode_grid(cache), projector(tokens), atol=2e-6, rtol=2e-5)


def test_masks_do_not_read_missing_groups_and_reject_partial_packets():
    projector = SharedProgressiveTokenAggregator(24, 4, 32, (2, 3), "float64", 8)
    tokens = torch.randn(2, 32, 24, requires_grad=True)
    mask = prefix_mask(torch.tensor([8, 24]), 32)
    actual = projector(tokens, mask)
    for sample, k in enumerate((8, 24)):
        torch.testing.assert_close(actual[sample:sample + 1], projector(tokens[sample:sample + 1, :k]))
    gradient = torch.autograd.grad(actual[0].square().mean(), tokens)[0]
    assert torch.count_nonzero(gradient[0, 8:]) == 0
    assert gradient[0, :8].norm() > 0
    with pytest.raises(ValueError, match="packet"):
        projector(tokens[:, :9])


def test_cache_rejects_bad_appends_before_mutation_and_content_changes_output():
    projector = SharedProgressiveTokenAggregator(24, 4, 32, (2, 3), "float64", 8).eval()
    tokens = torch.randn(1, 32, 24)
    cache = projector.new_cache(1)
    projector.append(cache, tokens[:, :8], 0)
    reference = projector.decode_grid(cache).clone()
    for start, value in [(0, tokens[:, :8]), (8, tokens[:, :9]), (8, tokens)]:
        with pytest.raises(ValueError):
            projector.append(cache, value, start)
        assert cache.tokens == 8
        torch.testing.assert_close(projector.decode_grid(cache), reference, atol=0, rtol=0)
    changed = tokens.clone()
    changed[:, 8:] = tokens[:, :1]
    assert (projector(changed) - projector(tokens)).norm() > 0.001
    projector.train()
    with pytest.raises(RuntimeError, match="inference"):
        projector.append(cache, tokens[:, 8:16], 8)


def test_architecture_migration_enumerates_new_parameters_and_rejects_unmatched_source(codec):
    from dataclasses import asdict, replace

    from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec
    from cp_adaptok.runtime import migrate_global_aggregation

    artifact = {"model_config": asdict(codec.config), "model": codec.state_dict()}
    target = CausalProgressiveCodec(replace(codec.config, token_aggregation="shared_progressive_v2"))
    report = migrate_global_aggregation(target, artifact)
    assert report["new_parameters"] == ["decoder.projector.gate.bias", "decoder.projector.gate.weight"]
    assert report["optimizer_restored"] is False
    for name, tensor in codec.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor, atol=0, rtol=0)
    artifact["model_config"]["n_max"] = 64
    with pytest.raises(ValueError, match="architecture"):
        migrate_global_aggregation(target, artifact)


def test_single_packet_aggregation_is_the_original_attention():
    global_attention = TokenGridProjector(24, 4, 8, (2, 3), "float64")
    progressive = SharedProgressiveTokenAggregator(24, 4, 8, (2, 3), "float64", 8)
    weights = progressive.state_dict()
    weights.update(global_attention.state_dict())
    progressive.load_state_dict(weights, strict=True)
    tokens = torch.randn(2, 8, 24)
    torch.testing.assert_close(progressive(tokens), global_attention(tokens), atol=0, rtol=0)
