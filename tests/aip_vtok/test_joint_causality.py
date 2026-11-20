import pytest
import torch
import torch.nn.functional as F

from aip_vtok.c1_followup.quantization import ORIGINAL
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.joint.bitstream import FRAME_HEADER
from aip_vtok.joint.codec import JointCodec
from aip_vtok.joint.scan import AREA, HEIGHT, SLOT_COUNT, WIDTH


class ToyBackbone(torch.nn.Module):
    def begin_frame(self, state, frame_id):
        intra = state.is_intra(frame_id, 32)
        context = None if intra else F.interpolate(state.feature.mean(1, keepdim=True), scale_factor=2).expand(-1, 256, -1, -1)
        return FrameContext(frame_id, intra, context, state.feature)

    def aligned_latent(self, x, context):
        return x

    def decode_aligned(self, latent, context, checkpointed=False):
        return latent[:, :3], None


class ToyProjector(torch.nn.Module):
    def forward(self, latent, intra):
        return latent[:, :1].expand(-1, 192 if intra else 256, -1, -1).contiguous()


def toy_codec():
    return JointCodec(ToyBackbone(), ToyProjector(), torch.arange(128).repeat(2, 1),
                      torch.tensor([[ORIGINAL] * 8] * 2), torch.ones(2, 8, 6), torch.arange(SLOT_COUNT)).eval()


def test_gop_boundary_and_reference_floor_do_not_depend_on_display():
    x = torch.randn(1, 128, HEIGHT, WIDTH)
    codec = toy_codec()
    profile = codec.profile()
    tx, rx = ReferenceState(frame_id=30, feature=torch.zeros(1, 256, HEIGHT, WIDTH)), None
    rx = ReferenceState(tx.frame_id, tx.feature.clone())
    for frame in (31, 32, 33):
        item = codec.encode_frame(x, tx, frame, profile)
        a = codec.begin_frame(tx, item.wire.data[:FRAME_HEADER.size], profile)
        b = codec.begin_frame(rx, item.wire.data[:FRAME_HEADER.size], profile)
        for session in (a, b):
            with pytest.raises(ValueError, match="reference"):
                session.commit_reference()
        a.decode_prefix(item.wire.prefix(a.reference_budget))
        b.decode_prefix(item.wire.prefix(b.reference_budget))
        saved = b.render(510).clone()
        tx, rx = a.commit_reference(), b.commit_reference()
        fingerprint = rx.fingerprint()
        assert tx.fingerprint() == fingerprint
        with pytest.raises(RuntimeError, match="already"):
            b.commit_reference()
        b.decode_prefix(item.wire.prefix(4096))
        assert torch.equal(saved, b.render(510))
        assert not torch.equal(saved, b.render(4096))
        assert rx.fingerprint() == fingerprint
        assert item.ordered_ids[:, :b.reference_budget].cpu().tolist()[0] == b.ids[:b.reference_budget]
        assert profile.table.shape == (8 * AREA,)


def test_receiver_rejects_repeated_or_shuffled_packets_without_mutation():
    codec = toy_codec()
    x = torch.zeros(1, 128, HEIGHT, WIDTH)
    profile = codec.profile()
    item = codec.encode_frame(x, ReferenceState(), 0, profile)
    a = codec.begin_frame(ReferenceState(), item.wire.data[:FRAME_HEADER.size], profile)
    one, two, *_ = profile.wire.packets(item.wire.prefix(510))
    with pytest.raises(ValueError, match="offset"):
        a.append_packet(two)
    assert len(a.ids) == 0
    a.append_packet(one)
    with pytest.raises(ValueError, match="offset"):
        a.append_packet(one)
    assert len(a.ids) == 128


def test_wrong_fsq_profile_and_model_version_reject_same_frame_header():
    codec = toy_codec()
    profile = codec.profile()
    header = bytearray(profile.wire.encode_frame(0, [0] * SLOT_COUNT).data[:FRAME_HEADER.size])
    with torch.no_grad():
        codec.fsq.log_scale[0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="identity"):
        codec.begin_frame(ReferenceState(), header, codec.profile())
    header[4] ^= 1
    with pytest.raises(ValueError, match="version"):
        codec.begin_frame(ReferenceState(), header, profile)
