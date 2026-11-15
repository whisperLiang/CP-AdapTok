import pytest
import torch

from cp_adaptok.decoder.token_grid_projector import TokenGridProjector
from cp_adaptok.tokenizer.learned_queries import LearnedQueryTokenizer
from cp_adaptok.tokenizer.prefix_masking import prefix_mask, sample_budgets


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_cached_softmax_matches_all_prefixes_and_masks(precision):
    torch.manual_seed(3)
    p = TokenGridProjector(dim=24, heads=4, n_max=64, grid=(3, 4), precision=precision).eval()
    tokens = torch.randn(2, 64, 24)
    cache = p.new_cache(batch_size=2)
    for start in range(0, 64, 8):
        k = start + 8
        cached = p.append(cache, tokens[:, start:k], start)
        reference = p(tokens[:, :k])
        masked = p(tokens, mask=prefix_mask(torch.tensor([k, k]), 64))
        torch.testing.assert_close(cached, reference, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(masked, reference, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        p.append(cache, tokens[:, :8], 0)


def test_extreme_softmax_is_stable():
    p = TokenGridProjector(dim=24, heads=4, n_max=16, grid=(2, 2)).eval()
    x = torch.randn(1, 16, 24) * 1000
    cache = p.new_cache(1)
    p.append(cache, x[:, :8], 0)
    y = p.append(cache, x[:, 8:], 8)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, p(x), atol=0.001, rtol=1e-4)


@pytest.mark.parametrize("n", [128, 2048, 4096, 8192])
def test_queries_are_one_ordered_full_sequence(n):
    model = LearnedQueryTokenizer(n_max=n, dim=24, heads=4, layers=2, kv_grid=(2, 3))
    y = torch.randn(1, 128, 4, 8)
    output = model(y, None, "P")
    assert output.shape == (1, n, 24)
    assert torch.equal(output[:, : min(n, 128)], model(y, None, "P")[:, : min(n, 128)])


def test_budget_sampling_covers_endpoints():
    rng = torch.Generator().manual_seed(5)
    samples = [sample_budgets((128, 256, 512, 1024, 2048), rng) for _ in range(50)]
    assert all(2 <= len(s) <= 4 and len(set(s)) == len(s) for s in samples)
    assert any(128 in s for s in samples) and any(2048 in s for s in samples)
