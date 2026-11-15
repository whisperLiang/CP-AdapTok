import torch

from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


@torch.no_grad()
def test_future_frames_and_all_enhancement_schedules_cannot_change_tokens(codec):
    torch.manual_seed(71)
    frames = [torch.rand(1, 3, 16, 24) for _ in range(3)]
    senders = [CodecEncoder(codec), CodecEncoder(codec)]
    receivers = [CodecDecoder(codec), CodecDecoder(codec)]
    for t in range(3):
        encoded = [s.encode_frame(frames[t]) for s in senders]
        assert encoded[0].packets == encoded[1].packets
        for i, (receiver, wire) in enumerate(zip(receivers, encoded)):
            receiver.start_frame(wire.header_bytes)
            receiver.append_packets(wire.packets[:1])
            receiver.decode_base()
            if i:
                receiver.append_packets(wire.packets[1:])
                receiver.decode_prefix(32)
        assert receivers[0].get_state().fingerprint() == receivers[1].get_state().fingerprint()
    a, b = CodecEncoder(codec), CodecEncoder(codec)
    first_a, first_b = a.encode_frame(frames[0]), b.encode_frame(frames[0])
    b.encode_frame(1 - frames[1])
    assert first_a.packets == first_b.packets
