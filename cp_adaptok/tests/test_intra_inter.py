import torch

from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.receiver import CodecDecoder


@torch.no_grad()
def test_periodic_intra_resets_memory_and_progressive_decodes(codec):
    encoder, decoder = CodecEncoder(codec, intra_period=2), CodecDecoder(codec)
    for t in range(4):
        x = torch.rand(1, 3, 31, 47)
        encoded = encoder.encode_frame(x)
        assert encoded.header.frame_type == ("I" if t % 2 == 0 else "P")
        decoder.start_frame(encoded.header_bytes)
        decoder.append_packets(encoded.packets)
        decoder.decode_base()
        assert decoder.decode_prefix(32).shape == x.shape
        assert decoder.get_state().fingerprint() == encoder.get_state().fingerprint()
        if t % 2 == 0:
            assert decoder.get_state().memory is None
