import torch

from hla_tok.uf_v2.native_weights import NativeAdjustments


def test_initial_native_adjustment_is_identity_with_live_gradients():
    model = NativeAdjustments()
    assert torch.all(model.encoder_gain() == 1)
    assert torch.all(model.reconstruction_gain() == 1)
    assert torch.all(model.hyper_bias() == 0)
    digest = model.digest()
    loss = model.encoder_gain().mean() + model.reconstruction_gain().mean() + model.hyper_bias().mean()
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.5).step()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert model.digest() != digest


def test_lite_quantization_profile_uses_bounded_and_canonical_configuration():
    import pytest

    from hla_tok.uf_v2.native_weights import LiteProfile

    assert LiteProfile(1, 1).identity() == LiteProfile(1.0, 1.0).identity()
    assert LiteProfile(1.4, .85).identity() != LiteProfile(1.0, 1.0).identity()
    for pair in ((0.9, 1), (4.1, 1), (2, .3), (2, 1.1)):
        with pytest.raises(ValueError, match="unsupported Base-Lite"):
            LiteProfile(*pair)
