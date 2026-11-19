import pytest
import torch

from aip_vtok.decoder.latent_canvas import LatentCanvas
from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer


@torch.no_grad()
def test_id_zero_arrives_and_incremental_equals_dense_without_reprojection():
    torch.manual_seed(9)
    tokenizer = ResidualTokenizer()
    predicted = torch.randn(1, 128, 8, 16)
    encoded = tokenizer.encode(torch.randn_like(predicted), predicted)
    ids = torch.zeros_like(encoded.ids)
    incremental = LatentCanvas(tokenizer, encoded.plan, predicted)
    for offset in range(0, ids.shape[1], 7):
        incremental.append_ids(ids[:, offset : offset + 7], offset)
    assert incremental.projected_tokens == ids.shape[1]
    assert incremental.masks[0].all()
    codes = tokenizer.fsq.ids_to_codes(ids)
    dense = predicted.clone()
    for pass_id in range(3):
        selected = encoded.plan.pass_ids == pass_id
        dense += tokenizer.synthesize_pass(
            codes[:, selected], encoded.plan.positions[selected], pass_id, (8, 16)
        )
    assert torch.equal(incremental.latent(), dense)
    with pytest.raises(ValueError, match="duplicate"):
        incremental.append_ids(ids[:, :1], 0)


def test_canvas_training_gradient_reaches_each_residual_projection():
    tokenizer = ResidualTokenizer()
    predicted = torch.zeros(1, 128, 8, 16)
    encoded = tokenizer.encode(torch.randn_like(predicted), predicted)
    canvas = LatentCanvas(tokenizer, encoded.plan, predicted)
    for offset in range(0, encoded.codes.shape[1], 128):
        canvas.append_codes(encoded.codes[:, offset : offset + 128], offset)
    canvas.latent().square().mean().backward()
    for layer in (tokenizer.anchor, tokenizer.innovation, tokenizer.detail):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())


def test_integer_append_remains_in_place_when_caller_has_gradients_enabled():
    tokenizer = ResidualTokenizer()
    prediction = torch.zeros(1, 128, 8, 16)
    tokens = tokenizer.encode(torch.randn_like(prediction), prediction)
    canvas = LatentCanvas(tokenizer, tokens.plan, prediction)
    pointer = canvas.buffers[0].data_ptr()
    canvas.append_ids(torch.zeros(1, 7, dtype=torch.long), 0)
    assert canvas.buffers[0].data_ptr() == pointer
    assert canvas.count == canvas.projected_tokens == 7
