import pytest
import torch

from aip_vtok.backbones.latent_to_ref import LatentToRef


@pytest.mark.parametrize("intra,channels", [(True, 192), (False, 256)])
def test_projector_has_native_frame_type_shapes_and_gradients(intra, channels):
    projector = LatentToRef(width=16)
    latent = torch.randn(1, 128, 4, 8, requires_grad=True)
    feature = projector(latent, intra)
    assert feature.shape == (1, channels, 8, 16)
    feature.square().mean().backward()
    assert latent.grad.abs().sum() > 0
    assert torch.isfinite(latent.grad).all()
