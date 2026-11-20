import pytest
import torch

from aip_vtok.c1_followup.quantization import ORIGINAL, Stage
from aip_vtok.joint.tokenizer import ProgressiveFSQ


def test_train_hard_path_matches_a2_and_integer_decode():
    levels = torch.tensor([[ORIGINAL, (10, 10, 8, 4, 5, 4)] * 4] * 2)
    scales = torch.rand(2, 8, 6) + 0.5
    fsq = ProgressiveFSQ(levels, scales)
    x = torch.randn(1, 8, 3, 4, 6, requires_grad=True)
    result = fsq.encode_train(x, True)
    groups = torch.arange(8).repeat_interleave(12)
    flat = result.ids.reshape(1, -1)
    old = Stage(levels[0].tolist(), scales[0])
    expected, saturation = old.encode(x.detach())
    assert torch.equal(result.ids, expected)
    assert torch.equal(result.saturated, saturation)
    hard = fsq.decode_ids(flat, groups, True).reshape_as(x)
    assert torch.equal(result.values, hard)
    assert torch.equal(hard, old.decode(expected))
    result.values.square().mean().backward()
    for tensor in (x.grad, fsq.log_scale.grad):
        assert tensor.isfinite().all() and tensor.abs().sum() > 0
    with pytest.raises(ValueError, match="64000"):
        fsq.decode_ids(torch.tensor([[64000]]), torch.tensor([0]), True)


def test_mixed_radix_covers_all_uint16_symbols():
    fsq = ProgressiveFSQ(torch.tensor([[ORIGINAL] * 8] * 2), torch.ones(2, 8, 6))
    ids = torch.arange(64000).view(1, -1)
    digits = fsq.digits(ids, torch.zeros(64000, dtype=torch.long), False)
    assert torch.equal((digits * fsq.radix[1, 0]).sum(-1), ids)
    assert fsq.decode_ids(ids, torch.zeros(64000, dtype=torch.long), False).isfinite().all()
