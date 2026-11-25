import struct

import pytest
import torch

from hla_tok.streaming.continuous import MAX_K, TOKEN_DIM, Order
from hla_tok.streaming.continuous_wire import (
    HEADER_BYTES,
    FrameReceiver,
    WireDtype,
    deserialize_continuous_tokens,
    max_budget_for_bytes,
    serialize_continuous_tokens,
    serialize_enhancement,
    wire_rate,
)


@pytest.fixture(scope="module")
def vectors():
    data = torch.arange(MAX_K * TOKEN_DIM, dtype=torch.float32).reshape(1, MAX_K, TOKEN_DIM)
    return (data % 1024 - 512) / 137


FORMATS = [(WireDtype.FP32, torch.float32, 4), (WireDtype.FP16, torch.float16, 2),
           (WireDtype.BF16, torch.bfloat16, 2)]


@pytest.mark.parametrize(("wire", "dtype", "width"), FORMATS)
def test_wire_roundtrip_bits_dimensions_and_actual_lengths(vectors, wire, dtype, width):
    values = vectors[:, :136]
    packet = serialize_continuous_tokens(values, wire, 417, order=Order.O2)
    decoded = deserialize_continuous_tokens(packet)
    assert (decoded.frame_id, decoded.token_offset, decoded.order) == (417, 0, Order.O2)
    assert decoded.dtype == wire and decoded.values.dtype == dtype
    assert decoded.values.shape == (1, 136, 32)
    assert torch.equal(decoded.values.view(torch.uint8), values.to(dtype).contiguous().view(torch.uint8))
    assert len(packet) == HEADER_BYTES + 136 * 32 * width


def test_fp32_uses_little_endian_and_preserves_ieee_bits():
    bits = (0x80000000, 0x3F800000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x00000001)
    pattern = b"".join(struct.pack("<I", word) for word in (bits * 6)[:32])
    floats = torch.tensor([(bits * 6)[:32]], dtype=torch.uint32).view(torch.float32).reshape(1, 1, 32)
    packet = serialize_continuous_tokens(floats, WireDtype.FP32, 0)
    assert packet[HEADER_BYTES:] == pattern
    assert bytes(deserialize_continuous_tokens(packet).values.view(torch.uint8).flatten().tolist()) == pattern


@pytest.mark.parametrize("wire", list(WireDtype))
def test_actual_payload_is_a_strict_prefix(vectors, wire):
    base = serialize_continuous_tokens(vectors[:, :510], wire, 42)
    enhanced = serialize_continuous_tokens(vectors[:, :12240], wire, 42)
    assert base[HEADER_BYTES:] == enhanced[HEADER_BYTES:len(base)]
    assert base[:HEADER_BYTES] != enhanced[:HEADER_BYTES]


def test_enhancement_contains_only_the_unpublished_suffix_and_preserves_prefix(vectors):
    fmt = WireDtype.BF16
    full = serialize_continuous_tokens(vectors[:, :136], fmt, 71)
    base = serialize_continuous_tokens(vectors[:, :1], fmt, 71)
    extension = serialize_enhancement(vectors, fmt, 71, 1, 136)
    assert len(base) + len(extension) == len(full) + HEADER_BYTES
    assert deserialize_continuous_tokens(extension).token_offset == 1

    receiver = FrameReceiver(71, fmt, Order.O0)
    assert receiver.accept(base) == 1
    earlier = receiver.prefix()
    assert receiver.accept(extension) == 136
    assert torch.equal(earlier, receiver.prefix()[:, :1])
    assert torch.equal(receiver.commit(), deserialize_continuous_tokens(full).values)


def test_gap_duplicate_overlap_and_late_commit_rejected(vectors):
    fmt = WireDtype.FP16
    receiver = FrameReceiver(3, fmt, Order.O1)
    ahead = serialize_enhancement(vectors, fmt, 3, 135, 136, order=Order.O1)
    base = serialize_continuous_tokens(vectors[:, :134], fmt, 3, order=Order.O1)
    assert receiver.accept(ahead) == 0
    assert receiver.prefix() is None
    assert receiver.accept(base) == 134
    with pytest.raises(ValueError, match="overlap"):
        receiver.accept(base)
    coverage = receiver.commit()
    assert coverage.shape == (1, 134, 32)
    missing = serialize_enhancement(vectors, fmt, 3, 134, 135, order=Order.O1)
    with pytest.raises(RuntimeError, match="committed"):
        receiver.accept(missing)
    assert torch.equal(receiver.prefix(), coverage)


def test_buffered_extensions_join_after_missing_gap_filled(vectors):
    fmt = WireDtype.FP32
    receiver = FrameReceiver(17, fmt, Order.O0)
    for small, large, expected in [(136, 510, 0), (1, 136, 0), (0, 1, 510)]:
        assert receiver.accept(serialize_enhancement(vectors, fmt, 17, small, large)) == expected
    assert torch.equal(receiver.commit(), vectors[:, :510])


@pytest.mark.parametrize("position", [0, 4, 5, 7, 10, 12, 15, 25, 32, -1])
def test_corruption_and_truncated_packet_are_rejected(vectors, position):
    packet = serialize_continuous_tokens(vectors[:, :4], WireDtype.FP32, 1)
    damaged = bytearray(packet)
    damaged[position] ^= 1
    with pytest.raises(ValueError):
        deserialize_continuous_tokens(bytes(damaged))
    with pytest.raises(ValueError):
        deserialize_continuous_tokens(packet[:-1])
    with pytest.raises(ValueError):
        deserialize_continuous_tokens(packet + b"\x00")


def test_receivers_cannot_accept_another_frames_packet(vectors):
    packet = serialize_continuous_tokens(vectors[:, :1], WireDtype.FP32, 1, order=Order.O0)
    for frame, dtype, order in [(2, WireDtype.FP32, Order.O0), (1, WireDtype.BF16, Order.O0),
                                (1, WireDtype.FP32, Order.O1)]:
        receiver = FrameReceiver(frame, dtype, order)
        with pytest.raises(ValueError, match="identity"):
            receiver.accept(packet)


def test_wire_rejects_incorrect_range_and_batches(vectors):
    dtype = WireDtype.BF16
    with pytest.raises(ValueError):
        serialize_continuous_tokens(vectors[:, :2], dtype, 0, MAX_K - 1)
    with pytest.raises(ValueError):
        serialize_continuous_tokens(vectors[:, :2].expand(2, -1, -1), dtype, 0)
    with pytest.raises(ValueError):
        serialize_enhancement(vectors, dtype, 0, 15, 14)


def test_bpp_and_rate_charge_real_header_for_each_packet(vectors):
    base_count, enhanced_count = 510, 1020
    first = serialize_enhancement(vectors, WireDtype.BF16, 1, 0, base_count)
    second = serialize_enhancement(vectors, WireDtype.BF16, 1, base_count, enhanced_count)
    nbytes = len(first) + len(second)
    rate = wire_rate(nbytes - 2 * HEADER_BYTES, nbytes, 1)
    assert rate["header_bytes"] == 2 * HEADER_BYTES
    assert rate["payload_bpp"] == 8 * (enhanced_count * 64) / (1920 * 1080)
    assert rate["total_bpp"] == 8 * nbytes / (1920 * 1080)
    for fps in (24, 30, 60):
        assert rate[f"mbps_{fps}fps"] == 8 * nbytes * fps / 1e6


def test_budget_cap_uses_real_wire_size(vectors):
    fmt = WireDtype.FP32
    packet = serialize_continuous_tokens(vectors[:, :510], fmt, 1)
    assert max_budget_for_bytes(len(packet), fmt) == 510
    assert max_budget_for_bytes(len(packet) - 1, fmt) == 509
    assert max_budget_for_bytes(HEADER_BYTES, fmt) == 0
    assert max_budget_for_bytes(10_000_000, fmt) == MAX_K
