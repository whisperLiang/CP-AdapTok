import pytest
import torch

from cp_adaptok.backbones.intra_adapter import prepare_rgb, restore_rgb
from cp_adaptok.models.causal_progressive_codec import CodecConfig
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder


@pytest.mark.parametrize(
    "changes",
    [
        {"heads": 0},
        {"k_ref": 129},
        {"n_max": 16384},
        {"state_fingerprint": "unknown"},
        {"heads": 5},
        {"grid": (0, 2)},
    ],
)
def test_invalid_model_boundaries_raise_value_error(changes):
    with pytest.raises(ValueError):
        CodecConfig(**changes)


def test_padding_crop_and_color_round_trip():
    x = torch.rand(1, 3, 1080, 1920)
    centered = prepare_rgb(x)
    assert centered.shape == (1, 3, 1088, 1920)
    torch.testing.assert_close(restore_rgb(centered, 1080, 1920), x, atol=3e-7, rtol=1e-5)
    with pytest.raises(ValueError):
        prepare_rgb(torch.rand(1, 1, 32, 32))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native 1080p structural test")
@torch.no_grad()
def test_actual_upstream_1080p_ip_bytes_path(codec):
    # Explicitly random weights: this checks shapes/bitstream/state, never trained quality.
    import copy

    model = copy.deepcopy(codec).cuda().eval()
    sender, receiver = CodecEncoder(model), CodecDecoder(model)
    for _ in range(2):
        frame = torch.rand(1, 3, 1080, 1920, device="cuda")
        encoded = sender.encode_frame(frame)
        receiver.start_frame(encoded.header_bytes)
        receiver.append_packets(encoded.packets)
        receiver.decode_base()
        output = receiver.decode_prefix(32, use_cache=True)
        assert output.shape == frame.shape and torch.isfinite(output).all()
        assert receiver.get_state().fingerprint() == sender.get_state().fingerprint()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native prefix-cache regression")
@torch.no_grad()
def test_native_1080p_default_queries_cached_reconstruction_matches_reference():
    from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec, CodecConfig

    torch.manual_seed(20261115)
    model = CausalProgressiveCodec(CodecConfig()).cuda().eval()
    sender, receiver = CodecEncoder(model), CodecDecoder(model)
    for _ in range(2):
        encoded = sender.encode_frame(torch.rand(1, 3, 1080, 1920, device="cuda"))
        receiver.start_frame(encoded.header_bytes)
        receiver.append_packets(encoded.packets)
        receiver.decode_base()
        for k in (128, 256, 512, 1024, 2048):
            dense = receiver.decode_prefix(k, use_cache=False)
            cached = receiver.decode_prefix(k, use_cache=True)
            torch.testing.assert_close(cached, dense, atol=1e-4, rtol=1e-4)
