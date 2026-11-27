import torch

from baselines.hla_tok_uf_v2.native_weights import NativeAdjustments


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
