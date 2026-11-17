import pytest
import torch

from experiments.original_adaptok_1080p.model_adapter import validate_budget
from experiments.original_adaptok_1080p.reconstruction import reconstruct_window
from experiments.original_adaptok_1080p.tiling import TileLayout


class IdentityAdapter:
    """Only a plumbing fixture: never used as evidence of official model quality."""
    device = torch.device("cpu")
    input_size = 4
    frame_num = 2

    def __init__(self):
        self.encodes = 0
        self.decodes = []

    def encode(self, x):
        self.encodes += 1
        return {"encoded": x}

    def decode(self, encoded, budget):
        self.decodes.append((id(encoded), budget))
        return encoded["encoded"]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_budgets_share_full_encoder_and_batching_preserves_pixels(batch_size):
    adapter = IdentityAdapter()
    clip = torch.rand(2, 3, 7, 11)
    plan = TileLayout(7, 11, 4, 4)
    results, timing = reconstruct_window(clip, adapter, plan, [512, 1024, 2048], batch_size)
    for output in results.values():
        assert torch.equal(output, clip)
    assert adapter.encodes == math_ceil(len(plan.coordinates), batch_size)
    for offset in range(0, len(adapter.decodes), 3):
        assert len({i for i, _ in adapter.decodes[offset:offset+3]}) == 1
    assert timing["tile_batch_size"] == batch_size


def math_ceil(n, d):
    return (n + d - 1) // d


@pytest.mark.parametrize("budget", [0, 513, 4096])
def test_invalid_group_budget_fails(budget):
    with pytest.raises(ValueError):
        validate_budget(budget, groups=4, tokens_per_group=512)
