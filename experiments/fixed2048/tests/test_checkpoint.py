from dataclasses import replace

import pytest
import torch

from cp_adaptok.models.causal_progressive_codec import CodecConfig
from experiments.fixed2048.checkpoint import migrate_grid, require_checkpoint, restore_rng, rng_state


def test_grid_migration_is_spatial_and_exhaustive():
    config = CodecConfig()
    state = {'decoder.projector.query': torch.arange(510 * 192).float().reshape(1, 510, 192),
             'other': torch.tensor([7.0])}
    target = replace(config, grid=(34, 60))
    migrated, ledger = migrate_grid(state, config, target)
    expected = torch.nn.functional.interpolate(state['decoder.projector.query'].transpose(1, 2).reshape(
        1, 192, 17, 30), size=(34, 60), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
    assert torch.equal(migrated['decoder.projector.query'], expected)
    assert torch.equal(migrated['other'], state['other'])
    assert {row['parameter'] for row in ledger} == set(state)
    assert [r['operation'] for r in ledger] == ['bilinear_2d', 'reuse']
    with pytest.raises(ValueError, match='separate'):
        migrate_grid(state, config, replace(target, kv_grid=(34, 60)))


def test_missing_checkpoint_cannot_create_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_checkpoint(tmp_path / 'missing.pt')


def test_rng_roundtrip_restores_next_draw():
    generator = torch.Generator().manual_seed(71)
    saved = rng_state(generator)
    expected = torch.rand(9, generator=generator)
    expected_global = torch.rand(9)
    restore_rng(saved, generator)
    assert torch.equal(torch.rand(9, generator=generator), expected)
    assert torch.equal(torch.rand(9), expected_global)
