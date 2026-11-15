import random
from dataclasses import replace

import pytest

from cp_adaptok.streaming.bitstream import FrameHeader, Packet, StreamError
from cp_adaptok.streaming.entropy_codec import ProgressiveEntropyCodec, quantize_cdf
from cp_adaptok.streaming.packetizer import ProgressivePacketStream, packetize


def header():
    return FrameHeader(
        frame_id=0,
        frame_type="I",
        reference_id=-1,
        height=1080,
        width=1920,
        n_max=256,
        k_ref=64,
        packet_tokens=64,
        codebook_size=32,
        model_id="11" * 16,
        codebook_id="22" * 16,
        entropy_id="33" * 16,
    )


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
    broken = bytearray(packets[0])
    broken[-1] ^= 1
    with pytest.raises(StreamError, match="CRC"):
        Packet.from_bytes(bytes(broken))


def test_cdf_exact_deterministic_positive():
    cdf = quantize_cdf([0, 1e-30, 100, 100])
    assert cdf == quantize_cdf([0, 1e-30, 100, 100])
    assert cdf[0] == 0 and cdf[-1] == 65536
    assert all(b > a for a, b in zip(cdf, cdf[1:]))


def test_malformed_valid_crc_packets_leave_the_prefix_unchanged():
    h = header()
    entropy = ProgressiveEntropyCodec.uniform(32)
    packets = packetize(h, [0] * h.n_max, entropy)
    first = Packet.from_bytes(packets[0])
    malformed = [
        replace(first, frame_id=1),
        replace(first, block_id=1),
        replace(first, token_count=63),
        replace(first, payload=b"x" * 200),
        replace(first, payload=b"\x00" * 4),
    ]
    stream = ProgressivePacketStream(h, entropy)
    for packet in malformed:
        with pytest.raises(StreamError):
            stream.append(packet.to_bytes())
        assert stream.token_ids == []
    for packet in packets:
        stream.append(packet)
    extra = Packet(h.frame_id, h.n_max, h.packet_tokens, 4, entropy.encode([0] * 64, 4))
    with pytest.raises(StreamError, match="N_max"):
        stream.append(extra.to_bytes())
    assert stream.token_ids == [0] * h.n_max
