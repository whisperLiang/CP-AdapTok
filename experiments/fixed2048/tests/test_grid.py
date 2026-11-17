from dataclasses import replace

import pytest
import torch

from cp_adaptok.decoder.prefix_latent_decoder import UnifiedPrefixLatentDecoder
from cp_adaptok.models.causal_progressive_codec import CodecConfig
from experiments.fixed2048.checkpoint import migrate_grid


def test_real_decoder_migration_strictly_loads_all_shapes_and_keeps_latent_shape():
    torch.set_num_threads(2)
    source = UnifiedPrefixLatentDecoder()
    target = UnifiedPrefixLatentDecoder(grid=(34,60))
    converted, ledger = migrate_grid({'decoder.'+k:v for k,v in source.state_dict().items()},
                                     CodecConfig(), replace(CodecConfig(), grid=(34,60)))
    target.load_state_dict({k.removeprefix('decoder.'):v for k,v in converted.items()}, strict=True)
    assert target.projector.query.shape == (1,2040,192)
    assert sum(r['operation'] != 'reuse' for r in ledger) == 1
    with torch.no_grad():
        result = target(torch.randn(1,2048,192), (68,120), 'P')
    assert result.shape == (1,128,68,120)
    bad = dict(converted)
    bad['decoder.projector.query'] = bad['decoder.projector.query'][:, :-1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        target.load_state_dict({k.removeprefix('decoder.'):v for k,v in bad.items()}, strict=True)
