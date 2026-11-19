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


@torch.no_grad()
def test_real_1080p_encode_once_reference_budget_invariance_and_exact_append(backbone):
    from aip_vtok.backbones.latent_to_ref import LatentToRef
    from aip_vtok.models.codec import AIPVTokCodec

    codec = AIPVTokCodec(backbone, LatentToRef(16).cuda()).cuda().eval()
    source = torch.rand(1, 3, 1080, 1920, device="cuda")
    initial = ReferenceState()
    encoded = codec.encode_frame(source, initial, 0)
    assert codec.encode_calls == 1
    assert encoded.tokens.ids.shape == (1, 8192)
    session = codec.prefix_session(encoded, initial, 2048)
    reference = session.commit_reference_once()
    assert session.rgb_decoder_calls == 0
    assert reference.latent.shape == (1, 128, 68, 120)
    encoded = codec.encode_frame(source, reference, 1)
    states = []
    for budget in (510, 2048, 8192):
        session = codec.prefix_session(encoded, reference, budget)
        states.append(session.commit_reference_once())
        assert session.rgb_decoder_calls == 0
    assert states[0].fingerprint() == states[1].fingerprint() == states[2].fingerprint()
    seven = codec.begin_frame(1, encoded.tokens.tiers, reference)
    for offset in range(0, 8192, 7):
        seven.append(encoded.tokens.ids[:, offset : min(offset + 7, 8192)], offset)
    assert torch.equal(seven.canvas.latent(), session.canvas.latent())
    a, b = seven.render_frame(), session.render_frame()
    assert torch.equal(a, b)
    assert seven.rgb_decoder_calls == session.rgb_decoder_calls == 1
