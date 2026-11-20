import pytest
import torch

from aip_vtok.joint.scan import BUDGETS, HEIGHT, SLOT_COUNT, WIDTH, SharedScan, coverage


def test_prefix_cover_and_frozen_table():
    scan = SharedScan(torch.arange(SLOT_COUNT))
    with torch.no_grad():
        scan.score.zero_()
    table = scan.table()
    assert torch.equal(table.sort().values, torch.arange(SLOT_COUNT))
    assert coverage(table[:510]).unique().numel() == 256
    assert (table[:510] < 4 * HEIGHT * WIDTH).all()
    assert table[:510].diff().min() > 0
    masks = [scan.mask(table, k) for k in BUDGETS]
    for k, mask in zip(BUDGETS, masks, strict=True):
        assert mask.sum() == k
        assert mask[table[:k]].all()
    (masks[0] * torch.arange(SLOT_COUNT)).mean().backward()
    assert scan.score.grad.isfinite().all() and scan.score.grad.abs().sum() > 0
    assert torch.equal(scan.freeze(), table)
    assert torch.equal(scan.table(), table)
    assert scan.record()["slot_count"] == SLOT_COUNT


def test_reject_duplicate_slots():
    with pytest.raises(ValueError, match="permutation"):
        SharedScan(torch.zeros(SLOT_COUNT, dtype=torch.long))
