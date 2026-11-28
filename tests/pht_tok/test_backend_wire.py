from dataclasses import replace
from struct import pack

import pytest

from pht_tok.bitstream import (
    HEADER_BYTES,
    NO_REFERENCE,
    RECORD,
    Chunk,
    ChunkHeader,
    CorruptStream,
    Mode,
    StreamReader,
)


def make_stream(mode):
    header = ChunkHeader(b"12345678", 8, 0, 0, 1080, 1920, 32, 1, mode, 1, False)
    records = tuple(bytes([stage]) * (8 + stage) for stage in range(9))
    return Chunk.pack(header, records), records


@pytest.mark.parametrize("mode_name", ["LD_MASK", "MASK_FINE"])
def test_all_eight_prefixes_reuse_the_complete_stream(mode_name):
    mode = getattr(Mode, mode_name)
    packet, records = make_stream(mode)
    for i, boundary in enumerate(packet.prefix_boundaries):
        prefix = packet.prefix(max_bytes=boundary)
        assert prefix == packet.data[:boundary]
        reader = StreamReader()
        assert reader.append_bytes(prefix[:7]) == ()
        assert reader.append_bytes(prefix[7:]) == records[:i + 2]
        assert reader.at_boundary
        assert reader.header == ChunkHeader.unpack(packet.data[:HEADER_BYTES])
    assert len(packet.prefix_boundaries) == 8


def test_index_and_dependency_record_mutations_are_rejected():
    packet, _ = make_stream(Mode.LD_MASK)
    altered = (packet.prefix_boundaries[0] - 1, *packet.prefix_boundaries[1:])
    with pytest.raises(ValueError, match="prefix index"):
        replace(packet, prefix_boundaries=altered)
    with pytest.raises(ValueError, match="record lengths"):
        replace(packet, sizes=(packet.sizes[0] - 1, *packet.sizes[1:]))
    with pytest.raises(ValueError, match="exceeds"):
        replace(packet, data=packet.data[:-1])
    first = HEADER_BYTES + RECORD.size + packet.sizes[0]
    second, third = packet.prefix_boundaries[:2]
    swapped = packet.data[:first] + packet.data[second:third] + packet.data[first:second] + packet.data[third:]
    with pytest.raises(CorruptStream, match="checksum"):
        StreamReader().append_bytes(swapped)
    reader = StreamReader()
    with pytest.raises(CorruptStream, match="length"):
        reader.append_bytes(packet.data[:first] + pack(">I", 1) + packet.data[first + 4:])
    with pytest.raises(CorruptStream, match="invalidated"):
        reader.append_bytes(packet.data)


def test_partial_input_and_replayed_append_do_not_move_a_boundary():
    packet, contents = make_stream(Mode.LD_MASK)
    boundary = packet.prefix_boundaries[0]
    reader = StreamReader()
    assert reader.append_bytes(packet.data[:boundary + 3], offset=0) == contents[:2]
    assert reader.append_bytes(packet.data[:boundary], offset=0) == ()
    with pytest.raises(CorruptStream, match="replay"):
        reader.append_bytes(b"X" + packet.data[1:boundary], offset=0)
    assert reader.append_bytes(packet.data[boundary + 3:], offset=boundary + 3) == contents[2:]
    assert reader.complete


def test_pht1_refresh_is_byte_stable_and_v2_geometry_is_backend_specific():
    header = ChunkHeader(b"12345678", 8, 0, 0, 1080, 1920, 32, 8, Mode.MASK, 1, False)
    assert header.pack()[:4] == b"PHT1"
    with pytest.raises(ValueError, match="dependencies"):
        replace(header, mode=Mode.LD_MASK)
    anchor = ChunkHeader(b"12345678", 0, NO_REFERENCE, 0, 1080, 1920, 32, 1, Mode.INTRA, 1, False)
    encoded = Chunk.pack(anchor, (b"four",))
    assert encoded.data[:4] == b"PHT1"
    assert encoded.prefix(max_bytes=len(encoded.data)) == encoded.data
    assert StreamReader().append_bytes(encoded.data) == (b"four",)
    with pytest.raises(CorruptStream, match="length"):
        ChunkHeader.unpack(encoded.data[:HEADER_BYTES - 1])
