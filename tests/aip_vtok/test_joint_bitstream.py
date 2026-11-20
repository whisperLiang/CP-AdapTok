import random

import pytest
import torch

from aip_vtok.c1_followup.quantization import ORIGINAL
from aip_vtok.joint.bitstream import FRAME_HEADER, PACKET_HEADER, WireFormat
from aip_vtok.joint.entropy import DigitEntropy
from aip_vtok.joint.scan import BUDGETS, SLOT_COUNT, SharedScan


def fixture_wire():
    levels = torch.tensor([[[10, 10, 8, 4, 5, 4]] * 4 + [list(ORIGINAL)] * 4] * 2)
    model = DigitEntropy(levels)
    with torch.no_grad():
        model.logits.normal_(std=2)
    return WireFormat(SharedScan(torch.arange(SLOT_COUNT)).table(), model.tables(), "11" * 32)


def test_rans_prefixes_and_every_independent_packet_round_trip():
    wire = fixture_wire()
    rng = random.Random(5)
    ids = [rng.randrange(64000) for _ in range(SLOT_COUNT)]
    for frame in (0, 1):
        result = wire.encode_frame(frame, ids)
        assert wire.parse_header(result.data[:FRAME_HEADER.size]) == frame
        packets = list(wire.packets(result.data))
        assert all(0 < count <= 128 for _, count in wire.packet_sizes.items())
        for start, payload in zip(reversed(wire.packet_sizes), reversed(packets), strict=True):
            offset, received = wire.decode_packet(frame, payload)
            assert offset == start and list(received) == ids[start:start + len(received)]
        for k in BUDGETS:
            prefix = result.prefix(k)
            assert prefix == result.data[:len(prefix)]
            ledger = result.ledger(k)
            assert ledger["total_bytes"] == len(prefix)
            assert len(prefix) == ledger["payload_bytes"] + ledger["header_bytes"]
            assert ledger["actual_bpp"] == 8 * len(prefix) / (1920 * 1080)


def test_reject_out_of_range_id_wrong_header_and_corrupt_or_truncated_packets():
    wire = fixture_wire()
    ids = [7] * SLOT_COUNT
    ids[303] = 64000
    with pytest.raises(ValueError, match="FSQ ID"):
        wire.encode_frame(0, ids)
    ids[303] = 0
    data = wire.encode_frame(0, ids).data
    header = bytearray(data[:FRAME_HEADER.size])
    header[-1] ^= 1
    with pytest.raises(ValueError, match="scan"):
        wire.parse_header(header)
    first = next(wire.packets(data))
    corrupt = bytearray(first)
    corrupt[PACKET_HEADER.size + 1] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        wire.decode_packet(0, corrupt)
    with pytest.raises(ValueError, match="truncated"):
        list(wire.packets(data[:-1]))
