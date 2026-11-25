import numpy as np
import torch

from hla_tok.uf.refinement import GroupLayout, RefinementQuantizer, priority_order


def test_nested_quantizer_improves_and_has_finite_gradient():
    quantizer = RefinementQuantizer(channels=3)
    latent = torch.linspace(-0.85, 0.85, 3 * 11 * 7).reshape(1, 3, 11, 7).requires_grad_()
    coarse, fine = quantizer(latent)
    symbol_values = quantizer.symbols(latent)
    assert all(values.dtype == torch.int8 for values in symbol_values)
    assert (latent - coarse - fine).square().mean() < (latent - coarse).square().mean()
    (latent - coarse - fine).square().mean().backward()
    assert quantizer.logit.grad is not None and quantizer.logit.grad.abs().sum() > 0
    assert torch.isfinite(quantizer.logit.grad).all()


def test_groups_cover_edge_without_coded_padding():
    layout = GroupLayout((7, 9, 11), tile=8, channels=4)
    symbols = np.arange(7 * 9 * 11, dtype=np.int32).reshape(layout.shape)
    restored = np.empty_like(symbols)
    for group in range(len(layout)):
        region = layout.region(group)
        restored[region] = symbols[region].flatten().reshape(layout.extent(group))
    np.testing.assert_array_equal(symbols, restored)
    assert sum(np.prod(layout.extent(group)) for group in range(len(layout))) == symbols.size


def test_order_is_dependency_ready_with_deterministic_ties():
    scores = np.array([[2., 3., 3.], [200., 1., 10.]])
    assert list(priority_order(scores, np.ones_like(scores, dtype=bool))) == [
        (0, 1), (0, 2), (1, 2), (0, 0), (1, 0), (1, 1),
    ]
