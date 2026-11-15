import torch

from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec, CodecConfig
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


@torch.no_grad()
def test_long_sequence_with_resets_and_different_enhancement_schedules(codec):
    sender = CodecEncoder(codec, intra_period=8)
    receivers = [CodecDecoder(codec), CodecDecoder(codec)]
    for t in range(32):
        encoded = sender.encode_frame(torch.rand(1, 3, 16, 24))
        for i, receiver in enumerate(receivers):
            receiver.start_frame(encoded.header_bytes)
            receiver.append_packets(encoded.packets[:1])
            receiver.decode_base()
            if i and t % 2:
                receiver.append_packets(encoded.packets[1:])
                receiver.decode_prefix(32, use_cache=True)
            assert receiver.get_state().fingerprint() == sender.get_state().fingerprint()


@torch.no_grad()
def test_requested_reference_prefix_ablation_values():
    torch.manual_seed(27)
    for k_ref in (128, 256, 512, 1024):
        config = CodecConfig(
            n_max=2048,
            k_ref=k_ref,
            dim=24,
            heads=4,
            codebook_size=32,
            embedding_dim=8,
            grid=(2, 3),
            kv_grid=(2, 3),
        )
        model = CausalProgressiveCodec(config).eval()
        sender, receiver = CodecEncoder(model), CodecDecoder(model)
        for _ in range(2):
            encoded = sender.encode_frame(torch.rand(1, 3, 16, 24))
            receiver.start_frame(encoded.header_bytes)
            receiver.append_packets(encoded.prefix_packets(k_ref))
            receiver.decode_base()
            assert sender.get_state().fingerprint() == receiver.get_state().fingerprint()
