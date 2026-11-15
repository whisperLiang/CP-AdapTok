import pytest
import torch

from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec, CodecConfig


@pytest.fixture(scope="session")
def codec():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    config = CodecConfig(
        n_max=32,
        k_ref=8,
        packet_tokens=8,
        dim=24,
        heads=4,
        codebook_size=32,
        embedding_dim=8,
        grid=(2, 3),
        kv_grid=(2, 3),
    )
    return CausalProgressiveCodec(config).eval()
