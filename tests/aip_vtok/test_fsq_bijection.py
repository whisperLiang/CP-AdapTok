import torch

from aip_vtok.tokenizer.fsq import FSQ


def test_all_64000_symbols_are_exactly_bijective_including_zero():
    fsq = FSQ()
    ids = torch.arange(64000)
    codes = fsq.ids_to_codes(ids)
    assert codes.shape == (64000, 6)
    assert torch.equal(fsq.codes_to_ids(codes), ids)
    assert torch.equal(fsq.digits_to_ids(fsq.ids_to_digits(ids)), ids)
    assert (codes[0] != 0).any()


def test_ste_gradient_is_finite_and_nonzero_with_even_level_bounds():
    fsq = FSQ()
    x = torch.randn(1024, 6, requires_grad=True)
    codes, ids = fsq(x)
    assert ids.min() >= 0 and ids.max() < 64000
    assert torch.equal(fsq.codes_to_ids(codes.detach()), ids)
    codes.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    extremes, _ = fsq(torch.tensor([[-100.0] * 6, [100.0] * 6]))
    assert torch.equal(fsq.codes_to_ids(extremes), torch.tensor([0, 63999]))
