import pytest
import torch

from hla_tok.uf_v2.decoder import SharedRefinement


@pytest.mark.parametrize("intra,latent_channels,feature_channels", [(True, 256, 384), (False, 128, 256)])
def test_refinement_zero_initialization_and_gradient(intra, latent_channels, feature_channels):
    torch.manual_seed(7)
    model = SharedRefinement()
    base = torch.randn(1, feature_channels, 8, 12)
    delta = torch.randn(1, latent_channels, 4, 6)
    current_context = None if intra else torch.randn(1, 256, 8, 12)
    result = model.refine(intra, base, delta, current_context)
    assert torch.equal(result, base)
    assert result.data_ptr() != base.data_ptr()
    result.square().mean().backward()
    assert model.branch(intra).out.weight.grad.abs().sum() > 0
    assert list(model.branch(not intra).parameters())[-1].grad is None


def test_inter_refinement_requires_same_frame_context():
    model = SharedRefinement()
    with pytest.raises(ValueError, match="context"):
        model.refine(False, torch.zeros(1, 256, 8, 8), torch.zeros(1, 128, 4, 4), None)
