from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cp_adaptok.models.causal_progressive_codec import CodecConfig
from cp_adaptok.tokenizer.vector_quantizer import AdapTokVQBottleneck
from experiments.fixed2048.codec import Fixed2048Reconstructor


class Intra(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def encode_latent(self, rgb, context):
        return rgb.mean(1, keepdim=True)[..., ::16, ::16].repeat(1, 256, 1, 1) * self.scale

    def decode_latent(self, latent, context):
        yuv = torch.nn.functional.interpolate(latent[:, :3], scale_factor=16)
        return yuv, yuv.mean(1, keepdim=True)


class Inter(Intra):
    def begin_frame(self, state):
        return state.feature, state.feature * 2

    def encode_latent(self, rgb, context):
        return super().encode_latent(rgb, context)[:, :128] + context.context.mean()

    def decode_latent(self, latent, context):
        yuv, _ = super().decode_latent(latent, context)
        return yuv, yuv.mean(1, keepdim=True) + context.context


class Tokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Parameter(torch.randn(1, 2048, 192) * 0.01)
        self.calls = 0

    def forward(self, y, context, frame_type):
        self.calls += 1
        return self.content + y.mean()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.lengths = []

    def forward(self, embeddings, shape, frame_type):
        self.lengths.append(embeddings.shape[1])
        return (embeddings.mean() * self.scale).expand(1, 256 if frame_type == "I" else 128, *shape)


def make_model():
    source = SimpleNamespace(
        config=CodecConfig(),
        intra=Intra(),
        ld=Inter(),
        tokenizer=Tokens(),
        vq=AdapTokVQBottleneck(),
        decoder=Decoder(),
    )
    return Fixed2048Reconstructor(source, identity="test")


def input_frame():
    return torch.rand(1, 3, 64, 64)


def test_pair_is_same_input_and_does_not_commit(monkeypatch):
    model = make_model().eval()
    encoded = model.encode(input_frame(), model.empty_state(), 0)
    outputs = model.compare(encoded, ("R0", "R0-spatial", "R1", "R2"))
    assert model.tokenizer.calls == 1
    assert not encoded.context.committed
    assert torch.equal(outputs["R0"].latent, encoded.latent)
    features = encoded.features
    z = model.vq.project_features(features)
    assert torch.allclose(outputs["R1"].embeddings, model.vq.project_out(z))
    distances = model.vq.distances(z.flatten(0, 1), model.vq.codebook_vectors())
    assert torch.equal(outputs["R2"].ids.flatten(), distances.argmin(-1))
    assert model.decoder.lengths == [2048, 2048]
    assert not hasattr(model, "scorer") and not hasattr(model, "entropy_model")


def test_continuous_never_reads_codebook(monkeypatch):
    model = make_model().eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("codebook was used")

    monkeypatch.setattr(model.vq, "forward", forbidden)
    monkeypatch.setattr(model.vq, "codebook_vectors", forbidden)
    result = model.reconstruct(model.encode(input_frame(), model.empty_state(), 0), "R1")
    assert result.ids is None
    assert result.rgb.shape == (1, 3, 64, 64)


def test_freezing_gradients_and_vq_training_policy():
    model = make_model().configure_training("R1").train()
    assert not model.intra.training and not model.ld.training
    encoded = model.encode(input_frame(), model.empty_state(), 0)
    result = model.reconstruct(encoded, "R1")
    result.rgb.mean().backward()
    for module in (model.tokenizer, model.vq.project_in, model.vq.project_out, model.decoder):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in model.intra.parameters())
    assert model.vq.codebook.weight.grad is None
    assert not model.vq.codebook.weight.requires_grad
    model.configure_training("R2")
    assert model.vq.codebook.weight.requires_grad
    assert model.vq.strategy == "deterministic_nearest"


def test_full_reference_and_independent_sequences():
    model = make_model().eval()
    encoded = model.encode(input_frame(), model.empty_state(), 0)
    output = model.reconstruct(encoded, "R1")
    state = model.advance(encoded, output, "full2048")
    assert state.feature is output.feature
    assert model.empty_state().feature is None
    second = model.encode(input_frame(), state, 1)
    assert second.context.frame_type == "P"
    assert second.latent.shape[1] == 128
    independent = model.encode(input_frame(), model.empty_state(), 1, force_intra=True)
    assert independent.context.frame_type == "I"
    assert independent.context.context is None
    with pytest.raises(RuntimeError, match="committed"):
        model.advance(encoded, output, "full2048")


def test_historical_reference_is_vq128_and_display_does_not_advance():
    model = make_model().eval()
    encoded = model.encode(input_frame(), model.empty_state(), 0)
    output = model.reconstruct(encoded, "R1")
    state = model.advance(encoded, output, "original128")
    assert model.decoder.lengths == [2048, 128]
    assert state.feature is not output.feature
    model.configure_training("R1").train()
    encoded = model.encode(input_frame(), model.empty_state(), 0)
    with pytest.raises(ValueError, match="diagnostic"):
        model.advance(encoded, model.reconstruct(encoded, "R1"), "original128")
