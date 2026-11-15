import random

import pytest

from cp_adaptok.streaming.entropy_codec import ProgressiveEntropyCodec, quantize_cdf
from cp_adaptok.streaming.bitstream import FrameHeader, Packet, StreamError
from cp_adaptok.streaming.packetizer import ProgressivePacketStream, packetize


def header():
    return FrameHeader(frame_id=0, frame_type="I", reference_id=-1, height=1080, width=1920,
                       n_max=256, k_ref=64, packet_tokens=64, codebook_size=32,
                       model_id="11" * 16, codebook_id="22" * 16, entropy_id="33" * 16)


@pytest.mark.parametrize("size", [2, 32, 4096, 8192])
def test_rans_round_trip(size):
    rng = random.Random(7)
    tokens = [rng.randrange(size) for _ in range(256)]
    codec = ProgressiveEntropyCodec([quantize_cdf([i + 1 for i in range(size)])])
    payload = codec.encode(tokens, 0)
    assert codec.decode(payload, len(tokens), 0) == tokens
    with pytest.raises(ValueError):
        codec.decode(payload + b"junk", len(tokens), 0)


def test_rans_uses_probability_distribution():
    codec = ProgressiveEntropyCodec([quantize_cdf([1000] + [1] * 31)])
    payload = codec.encode([0] * 256, 0)
    assert len(payload) < 30  # raw uint16 would take 512 bytes
    assert codec.decode(payload, 256, 0) == [0] * 256


def test_append_only_integrity_and_order():
    h = header()
    entropy = ProgressiveEntropyCodec.uniform(32)
    ids = [i % 32 for i in range(256)]
    packets = packetize(h, ids, entropy)
    assert FrameHeader.from_bytes(h.to_bytes()) == h
    assert h.padded_height == 1088
    stream = ProgressivePacketStream(h, entropy)
    with pytest.raises(StreamError, match="contiguous"):
        stream.append(packets[1])
    assert stream.token_ids == []
    stream.append(packets[0])
    prefix = h.to_bytes() + b"".join(packets[:1])
    for wire in packets[1:]:
        stream.append(wire)
    assert stream.token_ids == ids
    assert (h.to_bytes() + b"".join(packets)).startswith(prefix)
    with pytest.raises(StreamError):
        stream.append(packets[-1])
    broken = bytearray(packets[0]); broken[-1] ^= 1
    with pytest.raises(StreamError, match="CRC"):
        Packet.from_bytes(bytes(broken))


def test_cdf_exact_deterministic_positive():
    cdf = quantize_cdf([0, 1e-30, 100, 100])
    assert cdf == quantize_cdf([0, 1e-30, 100, 100])
    assert cdf[0] == 0 and cdf[-1] == 65536
    assert all(b > a for a, b in zip(cdf, cdf[1:]))
