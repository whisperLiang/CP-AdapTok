import torch

from cp_adaptok.tokenizer.vector_quantizer import AdapTokVQBottleneck


def test_deterministic_ids_embeddings_gradients_and_diagnostics():
    torch.manual_seed(4)
    vq = AdapTokVQBottleneck(input_dim=24, codebook_size=32, embedding_dim=16)
    x = torch.randn(2, 64, 24, requires_grad=True)
    result = vq(x)
    assert torch.equal(result.ids, vq(x).ids)
    torch.testing.assert_close(vq.embed(result.ids), result.embeddings)
    assert result.mask.all() and 1 <= result.perplexity <= 32.001
    assert 0 < result.usage <= 1
    (result.embeddings.square().mean() + result.loss).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert vq.codebook.weight.grad.abs().sum() > 0
