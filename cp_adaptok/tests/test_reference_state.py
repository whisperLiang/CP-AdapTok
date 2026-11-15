import pytest
import torch

from cp_adaptok.backbones.causal_state import CausalState
from cp_adaptok.streaming.bitstream import StreamError
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


@torch.no_grad()
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_enhancement_cannot_change_reference_and_peers_remain_synchronized(codec, precision):
    from dataclasses import replace

    from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec

    model = CausalProgressiveCodec(replace(codec.config, attention_precision=precision))
    model.load_state_dict(codec.state_dict(),strict=True)
    codec = model.eval()
    sender, receiver = CodecEncoder(codec), CodecDecoder(codec)
    torch.manual_seed(31)
    frames = [torch.rand(1, 3, 32, 48) for _ in range(4)]
    for t, x in enumerate(frames):
        wire = sender.encode_frame(x)
        receiver.start_frame(wire.header_bytes)
        receiver.append_packets(wire.packets[:1])
        base = receiver.decode_base()
        assert base.shape == x.shape
        fingerprint = receiver.get_state().fingerprint()
        assert fingerprint == sender.get_state().fingerprint()
        torch.testing.assert_close(receiver.get_state().feature, sender.get_state().feature, rtol=0, atol=0)
        if receiver.get_state().memory is not None:
            torch.testing.assert_close(receiver.get_state().memory, sender.get_state().memory, rtol=0, atol=0)
        receiver.append_packets(wire.packets[1:])
        receiver.decode_prefix(32, use_cache=True)
        assert receiver.get_state().fingerprint() == fingerprint
        assert receiver.context_builds == t + 1
        assert receiver.base_commits == t + 1
    assert codec.ld.model.memory is None and codec.ld.model.ctx is None  # upstream DPB is untouched


@torch.no_grad()
def test_base_loss_requires_recovery_model_mismatch_and_one_commit(codec):
    sender, receiver = CodecEncoder(codec), CodecDecoder(codec)
    first = sender.encode_frame(torch.rand(1, 3, 24, 32))
    receiver.start_frame(first.header_bytes)
    with pytest.raises(StreamError, match="reference prefix"):
        receiver.decode_base()
    second = sender.encode_frame(torch.rand(1, 3, 24, 32))
    with pytest.raises(StreamError):
        receiver.start_frame(second.header_bytes)
    receiver.reset()
    receiver.start_frame(first.header_bytes)
    receiver.append_packets(first.packets[:1])
    receiver.decode_base()
    receiver.decode_base()  # idempotent API; no second state commit
    assert receiver.base_commits == 1
    with pytest.raises(StreamError):
        receiver.start_frame(first.header_bytes)
    context = codec.begin_frame(CausalState.empty(codec.model_id), 0, "I", 24, 32)
    _, feature = codec.decode_ids(torch.zeros(1, 8, dtype=torch.long), context)
    context.commit(feature, "test")
    with pytest.raises(RuntimeError, match="already committed"):
        context.commit(feature, "test")
