import pytest
import torch


def test_13bit_roundtrip_including_endpoints():
    from adaptok_hr.codec.bitstream import pack_codes, unpack_codes

    for count in (1, 4, 7, 256, 512, 1024, 2048, 2685):
        ids = ([0, 8191, 1, 4096] * (count // 4 + 1))[:count]
        payload = pack_codes(ids)
        assert len(payload) == (13 * count + 7) // 8
        assert unpack_codes(payload, count) == ids
    for ids in ([-1], [8192], [1.5], [True]):
        with pytest.raises(ValueError):
            pack_codes(ids)
    with pytest.raises(ValueError):
        unpack_codes(b"\x00\x01", 1)  # nonzero padding bits


def test_real_bytes_roundtrip_nested_corruption_and_state(small_model):
    from adaptok_hr.codec.bitstream import HEADER, truncate_packet
    from adaptok_hr.codec.codec import Codec

    codec = Codec(small_model.eval())
    x = torch.rand(1, 3, 128, 256)
    packet, sending = codec.encode_frame(x, None, 32, stream_id="behavior-test")
    rgb, receiving = codec.decode_frame(packet, None)
    assert isinstance(packet, bytes) and len(packet) == HEADER.size + 52
    torch.testing.assert_close(sending.rgb, rgb, rtol=0, atol=0)
    torch.testing.assert_close(sending.feature, receiving.feature, rtol=0, atol=0)
    assert sending.chain == receiving.chain
    for budget in (4, 8, 16, 32):
        short = truncate_packet(packet, budget)
        decoded, _ = codec.decode_frame(short, None)
        ids = small_model.token_ids(x, None, budget)
        expected, _ = small_model.decode_ids(ids, None)
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    following, sending = codec.encode_frame(x, sending, 8)
    second, receiving = codec.decode_frame(following, receiving)
    torch.testing.assert_close(second, sending.rgb, rtol=0, atol=0)
    with pytest.raises(ValueError, match="state|reference|I frame"):
        codec.decode_frame(following, None)
    with pytest.raises(ValueError):
        codec.decode_frame(following, receiving)  # duplicate frame
    for index in (0, 5, len(packet) - 1):
        damaged = bytearray(packet)
        damaged[index] ^= 1
        with pytest.raises(ValueError):
            codec.decode_frame(bytes(damaged), None)
    with pytest.raises(ValueError):
        codec.decode_frame(packet[:-1], None)


def test_checkpoint_identity_is_enforced(small_model):
    from adaptok_hr.codec.codec import Codec

    x = torch.rand(1, 3, 128, 256)
    packet, _ = Codec(small_model.eval()).encode_frame(x, None, 4)
    with torch.no_grad():
        small_model.vq.codebook.weight[0, 0] += .1
    with pytest.raises(ValueError, match="identity|codebook|model"):
        Codec(small_model.eval()).decode_frame(packet, None)
