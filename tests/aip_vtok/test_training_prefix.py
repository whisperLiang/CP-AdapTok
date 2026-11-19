import torch

from aip_vtok.decoder.latent_canvas import LatentCanvas
from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer
from aip_vtok.training.prefix import prefix_canvas


@torch.no_grad()
def test_vectorized_training_prefix_is_exactly_the_same_as_incremental_inference():
    tokenizer = ResidualTokenizer()
    prediction = torch.randn(1, 128, 8, 16)
    tokens = tokenizer.encode(torch.randn_like(prediction), prediction)
    for budget in (7, 17, tokens.ids.shape[1]):
        incremental = LatentCanvas(tokenizer, tokens.plan, prediction)
        for offset in range(0, budget, 7):
            incremental.append_ids(tokens.ids[:, offset : min(offset + 7, budget)], offset)
        bulk = prefix_canvas(tokenizer, tokens, prediction, budget, differentiable=False)
        assert torch.equal(bulk.latent(), incremental.latent())
        assert all(torch.equal(a, b) for a, b in zip(bulk.masks, incremental.masks))
