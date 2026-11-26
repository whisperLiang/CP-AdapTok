import pytest

from pht_tok.bitstream import Chunk, ChunkHeader, CorruptStream, Mode, StreamReader


def example():
    header = ChunkHeader(b"model-id", 9, 8, 1, 1080, 1920, 32, 6, Mode.MASK, 2, True)
    layers = tuple(b"bytes" * count for count in range(1, 6))
    return Chunk.pack(header, layers), header, layers


def test_real_byte_boundaries_and_arbitrary_fragmentation():
    stream, header, layers = example()
    for stage, boundary in enumerate(stream.prefix_boundaries, 1):
        reader = StreamReader()
        prefix = stream.truncate_to_budget(boundary)
        assert prefix == stream.data[:boundary]
        received = [record for byte in prefix for record in reader.append_bytes(bytes([byte]))]
        assert reader.header == header
        assert tuple(received) == layers[:stage + 1]
        assert reader.at_boundary
        assert stream.truncate_to_budget(boundary - 1) == (
            b"" if stage == 1 else stream.data[:stream.prefix_boundaries[stage - 2]]
        )
    assert stream.ledger["physical_bytes"] == len(stream.data)
    assert stream.ledger["payload_bytes"] == list(map(len, layers))
    assert stream.ledger["layer_header_bytes"] + stream.ledger["header_bytes"] + sum(map(len, layers)) == (
        len(stream.data)
    )


def test_truncated_or_corrupted_layer_does_not_become_a_reference_boundary():
    stream, _, layers = example()
    reader = StreamReader()
    assert reader.append_bytes(stream.data[:stream.prefix_boundaries[1] - 1]) == layers[:2]
    assert not reader.at_boundary
    assert reader.append_bytes(stream.data[stream.prefix_boundaries[1] - 1:]) == layers[2:]
    assert reader.complete

    for damaged_offset in (4, stream.prefix_boundaries[0] - 1):
        damaged = bytearray(stream.data)
        damaged[damaged_offset] ^= 1
        with pytest.raises(CorruptStream):
            StreamReader().append_bytes(bytes(damaged))
    with pytest.raises(CorruptStream):
        StreamReader().append_bytes(stream.data + b"extra")
