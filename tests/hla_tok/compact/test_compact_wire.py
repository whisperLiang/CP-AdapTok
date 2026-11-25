from struct import pack

import pytest
import torch

from hla_tok.compression.packing import pack_signed, unpack_signed
from hla_tok.compression.profiles import Profile
from hla_tok.streaming.compact_wire import (
    HEADER_BYTES,
    CompactFrame,
    FrameAssembly,
    WireIdentity,
    deserialize,
    max_budget,
)
from hla_tok.streaming.continuous import MAX_K, Order


@pytest.mark.parametrize(
    ("bits", "values", "expected"),
    [(4, [-8, -1, 0, 7], b"\xf8\x70"),
     (6, [-32, -1, 0, 31], b"\xe0\x0f\x7c"),
     (8, [-128, -1, 0, 127], b"\x80\xff\x00\x7f")],
)
def test_signed_words_have_canonical_wire_bytes(bits, values, expected):
    q = torch.tensor(values, dtype=torch.int8).repeat(2, 1)
    payload = pack_signed(q, bits)
    assert payload == expected * 2
    assert torch.equal(unpack_signed(payload, bits, len(values)), q)


def test_unaligned_rows_pad_one_token_at_a_time():
    q = torch.tensor([[31, -32, -1], [1, -1, 15]], dtype=torch.int8)
    payload = pack_signed(q, 6)
    assert len(payload) == 6
    assert payload[:3] == pack_signed(q[:1], 6)
    assert torch.equal(unpack_signed(payload, 6, 3), q)
    with pytest.raises(ValueError, match="padding"):
        unpack_signed(payload[:2] + bytes([payload[2] | 0x80]) + payload[3:], 6, 3)


@pytest.fixture
def frame():
    identity = WireIdentity(Profile.Q8_6, Order.O2, b"a" * 32)
    values = torch.arange(80, dtype=torch.int8).remainder(61).reshape(10, 8) - 30
    return CompactFrame(identity, 57, pack_signed(values, 6))


def test_physical_packet_sizes_and_identity(frame):
    packet = frame.serialize_range(0, 7)
    assert len(packet) == HEADER_BYTES + 42
    parsed = deserialize(packet)
    assert parsed.identity == frame.identity
    assert parsed.frame_id == 57 and parsed.count == 7
    assert parsed.offset == 0
    assert parsed.payload == frame.payload[:42]
    assert parsed.ledger() == {"payload_bytes": 42, "header_bytes": HEADER_BYTES, "padding_bytes": 0,
                               "side_information_bytes": 0, "total_bytes": HEADER_BYTES + 42}
    assert max_budget(len(packet), frame.identity.profile) == 7
    assert max_budget(len(packet) - 1, frame.identity.profile) == 6
    assert max_budget(1, frame.identity.profile) == 0
    assert max_budget(10**8, frame.identity.profile) == MAX_K
    assert max_budget(26520, Profile.Q4_4, payload_only=True) == 13260


def test_incremental_bytes_missing_and_out_of_order(frame):
    base, middle, end = (frame.serialize_range(a, b) for a, b in [(0, 2), (2, 4), (4, 10)])
    receiver = FrameAssembly(frame.identity, frame.frame_id)
    assert receiver.accept(end) == 0
    assert receiver.prefix() is None
    assert receiver.accept(base) == 2
    assert receiver.prefix() == deserialize(base).payload
    with pytest.raises(ValueError, match="overlap"):
        receiver.accept(base)
    assert receiver.accept(middle) == 10
    assert receiver.commit() == frame.payload
    assert b"".join(bytes(deserialize(p).payload) for p in (base, middle, end)) == frame.payload
    with pytest.raises(RuntimeError, match="committed"):
        receiver.accept(end)
    missing = FrameAssembly(frame.identity, frame.frame_id)
    missing.accept(base)
    missing.accept(end)
    assert missing.commit() == frame.payload[:12]


def test_corrupted_metadata_and_profile_switch_are_rejected(frame):
    raw = frame.serialize_range(0, 5)
    for broken in (raw[:20], raw + b"x", raw[:-1], raw[:-1] + bytes([raw[-1] ^ 8]),
                   raw[:8] + pack("<I", 58) + raw[12:]):
        with pytest.raises(ValueError):
            deserialize(broken)
    for identity in (
        WireIdentity(Profile.Q8_6, Order.O1, frame.identity.signature),
        WireIdentity(Profile.Q8_6, Order.O2, b"b" * 32),
        WireIdentity(Profile.Q8_8, Order.O2, frame.identity.signature),
    ):
        with pytest.raises(ValueError, match="identity"):
            FrameAssembly(identity, 57).accept(raw)
    with pytest.raises(ValueError):
        frame.serialize_range(9, 11)
    with pytest.raises(ValueError):
        pack_signed(torch.tensor([[8]], dtype=torch.int8), 4)
