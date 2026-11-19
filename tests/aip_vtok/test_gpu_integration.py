"""Run explicitly: pytest -q tests/aip_vtok/test_gpu_integration.py."""

from pathlib import Path

import pytest
import torch

from aip_vtok.backbones.dcvc_aligned import AlignedDCVC, Geometry
from aip_vtok.decoder.causal_state import ReferenceState

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def backbone():
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA GPU")
    model = AlignedDCVC(
        ROOT / "checkpoints/cvpr2026_image.pth.tar", ROOT / "checkpoints/cvpr2026_video_ld.pth.tar"
    ).cuda()
    yield model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test_native_1080p_latent_geometry_and_single_reference(backbone):
    with torch.no_grad():
        rgb = torch.rand(1, 3, 1080, 1920, device="cuda")
        context = backbone.begin_frame(ReferenceState(), 0)
        y = backbone.native_latent(rgb, context)
        assert y.shape == (1, 256, 68, 120)
        rec, feature = backbone.decode_native(y, context)
        assert rec.shape == rgb.shape
        assert feature.shape == (1, 192, 136, 240)
        state = context.commit_reference_once(feature)
        context = backbone.begin_frame(state, 1)
        y = backbone.native_latent(rgb, context)
        rec, feature = backbone.decode_native(y, context)
        assert y.shape == (1, 128, 68, 120)
        assert feature.shape == (1, 256, 136, 240)
        assert torch.isfinite(rec).all()


def test_frozen_checkpointed_decoder_preserves_alignment_gradients(backbone):
    # Real pretrained decoder, reduced spatial crop for a quick invariant check.
    backbone.geometry = Geometry(64, 128)
    try:
        rgb = torch.rand(1, 3, 64, 128, device="cuda")
        context = backbone.begin_frame(ReferenceState(), 0)
        with torch.no_grad():
            native = backbone.native_latent(rgb, context)
        y = backbone.alignment.align(native, True)
        assert y.requires_grad
        rec, _ = backbone.decode_aligned(y, context, checkpointed=True)
        rec.square().mean().backward()
        gradient = backbone.alignment.intra_down.weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert all(p.grad is None for p in backbone.intra.parameters())
    finally:
        backbone.geometry = Geometry()
        backbone.zero_grad(set_to_none=True)
