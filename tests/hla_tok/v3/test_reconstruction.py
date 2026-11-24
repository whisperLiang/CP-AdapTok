from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hla_tok import HLATok
from hla_tok.models.spatial import DenseReconstruction, RGBDecoder, SpatialEncoder
from hla_tok.training.losses import ReconstructionLoss


def test_dense_reconstruction_has_only_the_coarse_bottleneck():
    codec = HLATok("R2")
    dense = DenseReconstruction(codec.encoder, codec.rgb_decoder)
    coarse = []

    def observe_input(module, args):
        coarse.append(args[0].shape)

    handle = dense.rgb_decoder.register_forward_pre_hook(observe_input)
    with patch.object(codec.query_encoder, "prepare", side_effect=AssertionError("Tokenizer entered")):
        with patch.object(codec.quantizer, "lookup", side_effect=AssertionError("VQ entered")):
            image = dense(torch.rand(1, 3, 128, 256))
    handle.remove()
    assert coarse == [(1, 128, 8, 16)]
    assert image.shape == (1, 3, 128, 256)
    assert set(dict(dense.named_children())) == {"encoder", "rgb_decoder"}
    image.mean().backward()
    assert codec.encoder.stem[1].weight.grad.count_nonzero()
    assert codec.rgb_decoder.upsample[-2].weight.grad.count_nonzero()


def test_d1_preserves_d0_function_at_initialization_and_legacy_keys():
    base = HLATok("R2")
    saved = base.state_dict()
    old = HLATok("R2", rgb_variant="D0")
    old.load_state_dict(saved, strict=True)
    expanded = RGBDecoder("D1")
    mismatches = expanded.load_state_dict(old.rgb_decoder.state_dict(), strict=False)
    assert not mismatches.unexpected_keys
    assert {name.rsplit(".", 2)[0] for name in mismatches.missing_keys} == {"upsample.2.2", "upsample.5.2"}
    source = torch.rand(1, 3, 128, 128)
    dense = DenseReconstruction(old.encoder, expanded)
    with torch.no_grad():
        result = dense(source)
        torch.testing.assert_close(result, DenseReconstruction(base.encoder, base.rgb_decoder)(source), rtol=0, atol=0)


@pytest.mark.gpu
def test_true_perceptual_terms_remain_fp32_and_differentiate_reconstruction():
    loss = ReconstructionLoss("cuda").train()
    assert not any(module.training for module in loss.modules())
    assert not any(param.requires_grad for param in loss.parameters())
    source = torch.rand(1, 3, 512, 512, device="cuda")
    predicted = torch.rand_like(source, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        terms = loss(source, predicted, ((0, 0),))
    for key in ("lpips_alex", "lpips_vgg", "dists"):
        assert terms[key].dtype == torch.float32
        gradient, = torch.autograd.grad(terms[key], predicted, retain_graph=True)
        assert gradient.isfinite().all() and gradient[..., -128:, -128:].count_nonzero()


@pytest.mark.gpu
def test_dense_native_coarse_and_rgb_shapes():
    dense = DenseReconstruction(SpatialEncoder(), RGBDecoder("D1")).cuda().to(memory_format=torch.channels_last)
    source = torch.rand(1, 3, 1080, 1920, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert dense.coarse(source)[0].shape == (1, 128, 72, 120)
        assert dense(source).shape == source.shape


def test_native_path_preserves_original_pixel_grid():
    dense = DenseReconstruction(SpatialEncoder(), RGBDecoder())
    source = torch.rand(1, 3, 1080, 1920)
    observed = []
    hook = dense.encoder.stem[0].register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    with torch.inference_mode(), patch.object(F, "interpolate", side_effect=AssertionError("RGB was resized")):
        decoded = dense(source)
    hook.remove()
    assert decoded.shape == source.shape
    assert torch.equal(observed[0], F.pad(source, (0, 0, 0, 72), mode="reflect"))


def test_alex_preprocessing_does_not_detach_reconstruction():
    class Perceptual(nn.Module):
        def forward(self, target, predicted):
            return (predicted - target).square().mean()

    loss = ReconstructionLoss("cpu", lpips_alex=0, lpips_vgg=0, dists=0)
    loss.alex = Perceptual()
    original = torch.zeros(1, 3, 512, 512)
    predicted = torch.full_like(original, 0.4, requires_grad=True)
    term = loss(original, predicted, ((0, 0),))["lpips_alex"]
    assert term.requires_grad
    grad, = torch.autograd.grad(term, predicted)
    assert term.item() == pytest.approx(0.64)
    assert grad.count_nonzero() == grad.numel()
