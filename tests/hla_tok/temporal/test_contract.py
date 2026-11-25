import numpy as np
import pytest
import torch

from hla_tok.streaming.continuous import MAX_K, NATIVE_RANKS, REGIONS, Order, RankOrder
from hla_tok.streaming.temporal_wire import FrameHeader, Mode, Record, open_header, open_record
from hla_tok.temporal.layout import NativeLayout
from hla_tok.temporal.quantization import quantize, reconstruct, round_divide
from hla_tok.temporal.sparse import pack_block, unpack_block


def test_native_layout_covers_each_real_token_once():
    layout = NativeLayout()
    native = np.arange(MAX_K)
    rank, region = layout.coordinates(native)
    assert np.array_equal(layout.index(rank, region), native)
    assert layout.mask.shape == (121, 135)
    assert layout.mask.sum() == MAX_K
    assert layout.mask[-1, :120].all()
    assert not layout.mask[-1, 120:].any()
    assert layout.index(120, 119) == 16319
    with pytest.raises(ValueError, match="native"):
        layout.index(120, 120)
    with pytest.raises(ValueError, match="native"):
        layout.coordinates(MAX_K)


@pytest.mark.parametrize("order", [RankOrder.native(), RankOrder.reversed(), RankOrder(Order.O2, NATIVE_RANKS[1:] + (0,))])
def test_layout_order_roundtrip(order):
    layout = NativeLayout()
    native = torch.arange(MAX_K, dtype=torch.float32)[None, :, None].expand(-1, -1, 32)
    wire = layout.order(native, order)
    assert torch.equal(wire[0, :, 0], torch.from_numpy(layout.wire_indices(order)).float())
    assert torch.equal(layout.restore(wire, order), native)
    assert len(set(wire[0, :, 0].tolist())) == MAX_K


def test_layout_recovery_arbitrary_rank_order_and_absent_tokens():
    layout = NativeLayout()
    mask = np.zeros(MAX_K, dtype=bool)
    mask[:REGIONS] = True  # A valid all-zero vector in each Region.
    mask[layout.index(5, 20)] = True
    mask[layout.index(3, 20)] = True
    mask[layout.index(120, 119)] = True
    native = torch.zeros((1, MAX_K, 32))
    native[0, layout.index(3, 20), 0] = 3
    native[0, layout.index(5, 20), 0] = 5
    native[0, layout.index(120, 119), 0] = 120
    groups = list(layout.recovery_groups(native, mask.tobytes()))
    assert [group.shape[1] for group, _ in groups] == [20, 1, 98, 1, 15]
    assert [ranks.tolist() for _, ranks in groups] == [[0], [0, 3, 5], [0], [0, 120], [0]]
    assert groups[1][0][0, 0, :, 0].tolist() == [0, 3, 5]
    assert groups[3][0][0, 0, :, 0].tolist() == [0, 120]


def test_integer_rounding_ties_and_clipping():
    delta = np.array([-19, -7, -5, -3, -1, 1, 3, 5, 7, 17, 255], dtype=np.int32)
    step = np.full(delta.shape, 2, dtype=np.int32)
    assert round_divide(delta, step).tolist() == [-10, -4, -2, -2, 0, 0, 2, 2, 4, 8, 128]
    prediction = np.full(delta.shape, 12, dtype=np.int16)
    target = prediction + delta
    codes, clipped = quantize(target, prediction, step, 4)
    assert codes.tolist() == [-8, -4, -2, -2, 0, 0, 2, 2, 4, 7, 7]
    assert clipped == 3
    assert np.array_equal(reconstruct(prediction, codes, step), prediction + 2 * codes.astype(np.int16))


@pytest.mark.parametrize("limit, code", [(32767, 1), (-32768, -1)])
def test_integer_overflow_is_rejected(limit, code):
    with pytest.raises(OverflowError):
        reconstruct(np.array([limit], dtype=np.int16), np.array([code], dtype=np.int8), np.array([1]))


@pytest.mark.parametrize(
    "symbols,bits,sparse,mode,payload,padding",
    [
        ([-8, -1, 7], 4, False, Mode.RAW4, b"\xf8\x07", 4),
        ([-128, -1, 127], 8, False, Mode.RAW8, b"\x80\xff\x7f", 0),
        ([0] * 9, 4, True, Mode.ZERO, b"", 0),
        ([0, -1, 0, 0, 0, 0, 0, 0, 7], 4, True, Mode.BITMAP4, b"\x02\x01\x7f", 7),
        ([-17, 0, 0], 8, True, Mode.RAW8, b"\xef\x00\x00", 0),
        ([1, 2, 3, 4], 8, True, Mode.RAW4, b"\x21\x43", 0),
    ],
)
def test_sparse_physical_mode_and_padding(symbols, bits, sparse, mode, payload, padding):
    values = np.array(symbols, dtype=np.int8)
    block = pack_block(values, bits, sparse)
    assert (block.mode, block.payload, block.padding_bits) == (mode, payload, padding)
    decoded, actual_padding = unpack_block(mode, payload, len(values))
    assert actual_padding == padding
    assert np.array_equal(decoded, values)


@pytest.mark.parametrize("mode,payload,count", [
    (Mode.RAW4, b"\xf2", 1), (Mode.ZERO, b"\x00", 1),
    (Mode.BITMAP4, b"\x80\x00", 3), (Mode.BITMAP4, b"\x01\x00", 3),
])
def test_sparse_rejects_noncanonical_payloads(mode, payload, count):
    with pytest.raises(ValueError):
        unpack_block(mode, payload, count)


def test_header_record_crc_and_wire_fields():
    asset, history = b"a" * 32, b"s" * 32
    header = FrameHeader(77, 76, 9, 31, asset, history)
    encoded = header.serialize()
    assert encoded.startswith(b"HLT\x01")
    assert open_header(encoded) == header
    packet = Record(3, 768, 4, False, Mode.RAW4, b"\xff\x00").serialize()
    record, stop = open_record(packet)
    assert record == Record(3, 768, 4, False, Mode.RAW4, b"\xff\x00")
    assert stop == len(packet)
    for index in (4, len(encoded) - 1):
        corrupt = bytearray(encoded)
        corrupt[index] ^= 1
        with pytest.raises(ValueError):
            open_header(bytes(corrupt))
    with pytest.raises(ValueError):
        open_record(packet[:-1] + bytes([packet[-1] ^ 1]))


def test_ordered_block_steps_other_than_two_round_to_nearest_integer():
    delta = np.array([0, 3, 5, 8, -2, -8, 4, -4], dtype=np.int32)
    step = np.array([3] * 6 + [7, 7], dtype=np.int32)
    assert round_divide(delta, step).tolist() == [0, 1, 2, 3, -1, -3, 1, -1]


@pytest.mark.parametrize("rank,region", [(-1, 0), (121, 0), (0, 135), (0, -1), (1.5, 3)])
def test_invalid_native_positions_raise(rank, region):
    with pytest.raises(ValueError, match="native"):
        NativeLayout().index(rank, region)


@pytest.mark.parametrize("value", [
    FrameHeader(2**32 - 1, 0, 3, 1, bytes(32), bytes(32)),
    FrameHeader(3, 2, 0, 1, bytes(32), bytes(32)),
    FrameHeader(3, 3, 2, 1, bytes(32), bytes(32)),
    FrameHeader(3, 2, 2, 0, bytes(32), bytes(32)),
    FrameHeader(3, 2, 2, 1, bytes(31), bytes(32)),
    FrameHeader(3, 2**32 - 1, 2, 1, bytes(32), b"a" * 32),
])
def test_invalid_headers_are_rejected_before_serialization(value):
    with pytest.raises(ValueError):
        value.serialize()


@pytest.mark.parametrize("data,bits", [(np.arange(2), 4), (np.zeros((2, 2), dtype=np.int8), 4),
                                     (np.zeros(257, dtype=np.int8), 8), (np.zeros(2, dtype=np.int8), 6),
                                     (np.array([-9, 1], dtype=np.int8), 4)])
def test_rejected_short_block_inputs(data, bits):
    with pytest.raises(ValueError):
        pack_block(data, bits, False)
