from pathlib import Path

import pytest
import torch

from aip_vtok.backbones.dcvc_aligned import AlignedDCVC, Geometry
from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.models.codec import AIPVTokCodec

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def codec():
    if not torch.cuda.is_available():
        pytest.skip("real CUDA required")
    backbone = AlignedDCVC(
        ROOT / "checkpoints/cvpr2026_image.pth.tar",
        ROOT / "checkpoints/cvpr2026_video_ld.pth.tar",
        geometry=Geometry(64, 128),
    ).cuda()
    yield AIPVTokCodec(backbone, LatentToRef(16).cuda()).cuda().eval()
    torch.cuda.synchronize()


@torch.no_grad()
def rollout(codec, clip):
    state, outputs = ReferenceState(), []
    for frame_id, rgb in enumerate(clip):
        encoded = codec.encode_frame(rgb[None], state, frame_id)
        session = codec.prefix_session(encoded, state, encoded.tokens.ids.shape[1])
        rgb = session.render_frame()
        state = session.commit_reference_once()
        outputs.append((encoded.tokens.ids.clone(), rgb.clone(), state.fingerprint()))
    return outputs


def test_changing_future_pixels_preserves_every_past_id_rgb_and_state(codec):
    clip = torch.rand(3, 3, 64, 128, device="cuda")
    changed = clip.clone()
    changed[2] = 1 - changed[2]
    a, b = rollout(codec, clip), rollout(codec, changed)
    for first, second in zip(a[:2], b[:2]):
        assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
        assert first[2] == second[2]


@torch.no_grad()
def test_frame_33_resets_prediction_and_reference_memory_without_rgb_commit(codec):
    rgb = torch.rand(1, 3, 64, 128, device="cuda")
    state, initial_ids = ReferenceState(), None
    for frame_id in range(33):
        encoded = codec.encode_frame(rgb, state, frame_id)
        session = codec.prefix_session(encoded, state, encoded.tokens.ids.shape[1])
        state = session.commit_reference_once()
        assert session.rgb_decoder_calls == 0
        if frame_id == 0:
            initial_ids = encoded.tokens.ids.clone()
        if frame_id == 32:
            assert torch.equal(encoded.tokens.ids, initial_ids)
            assert state.memory is None
            assert session.context.intra


@torch.no_grad()
def test_default_codec_has_no_external_predictor_in_64_frame_closed_loop(codec):
    assert codec.predictor is None
    rgb = torch.rand(1, 3, 64, 128, device="cuda")
    state = ReferenceState()
    for frame in range(64):
        encoded = codec.encode_frame(rgb, state, frame)
        session = codec.prefix_session(encoded, state, encoded.tokens.ids.shape[1])
        rec = session.render_frame()
        state = session.commit_reference_once()
        assert torch.isfinite(rec).all() and torch.isfinite(state.feature).all()
        assert state.latent.square().mean().sqrt() < 2
