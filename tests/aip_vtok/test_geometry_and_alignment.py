"""Gate 0 boundaries: real geometry and honest channel reduction."""

import pytest
import torch

from aip_vtok.backbones.dcvc_aligned import ChannelAlignment, Geometry, continuous_intra_latent


def test_native_padding_and_crop_restore_every_pixel():
    geometry = Geometry()
    rgb = torch.rand(1, 3, 1080, 1920)
    padded = geometry.pad(rgb)
    assert padded.shape == (1, 3, 1088, 1920)
    assert torch.equal(geometry.crop(padded), rgb)
    assert torch.equal(padded[..., 1080:, :], rgb[..., -1:, :].expand(-1, -1, 8, -1))


@pytest.mark.parametrize("shape", [(1, 3, 540, 960), (1, 1, 1080, 1920)])
def test_native_input_rejects_resize_and_wrong_channels(shape):
    with pytest.raises(ValueError, match="RGB"):
        Geometry().pad(torch.zeros(shape))


def test_continuous_i_scale_keeps_both_official_factors():
    y = torch.tensor([1.0, -3.0])
    assert torch.equal(continuous_intra_latent(y, torch.tensor(2.0), torch.tensor(3.0)), y * 6)


def test_i_pca_roundtrip_is_exact_only_on_representable_subspace():
    torch.manual_seed(11)
    alignment = ChannelAlignment()
    basis, _ = torch.linalg.qr(torch.randn(256, 128))
    samples = torch.randn(1, 128, 16, 32)
    latent = torch.einsum("ci,bihw->bchw", basis, samples)
    alignment.fit_intra([latent])
    recovered = alignment.restore(alignment.align(latent, True), True)
    torch.testing.assert_close(recovered, latent, atol=2e-5, rtol=2e-5)
    unrelated = torch.randn_like(latent)
    assert (alignment.restore(alignment.align(unrelated, True), True) - unrelated).square().mean() > 0.1


def test_p_alignment_normalizes_distribution_and_has_an_inverse():
    latent = torch.randn(1, 128, 16, 32) * 7 + 4
    alignment = ChannelAlignment()
    alignment.fit_predictive([latent])
    y = alignment.align(latent, False)
    torch.testing.assert_close(y.mean((0, 2, 3)), torch.zeros(128), atol=1e-5, rtol=0)
    torch.testing.assert_close(alignment.restore(y, False), latent, atol=1e-5, rtol=1e-5)


def test_alignment_gradient_exists_and_is_finite():
    alignment = ChannelAlignment()
    source = torch.randn(1, 256, 4, 8)
    y = alignment.align(source, True)
    alignment.restore(y, True).square().mean().backward()
    for parameter in alignment.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert alignment.intra_down.weight.grad.abs().sum() > 0
