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


def test_indices_select_the_nearest_known_code():
    vq = AdapTokVQBottleneck(input_dim=2, codebook_size=3, embedding_dim=2, normalized=False)
    with torch.no_grad():
        vq.project_in.weight.copy_(torch.eye(2))
        vq.project_in.bias.zero_()
        vq.codebook.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    result = vq(torch.tensor([[[0.9, 0.0], [0.0, 0.8], [-0.8, 0.0]]]))
    assert result.ids.tolist() == [[1, 2, 0]]
