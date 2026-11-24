import operator

import pytest

from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit


@pytest.mark.parametrize("ids", [[], [8191], [1, 2], list(range(8192))])
def test_fixed13_round_trip_and_byte_count(ids):
    payload = pack_tokens_13bit(ids)
    assert unpack_tokens_13bit(payload, len(ids)) == ids
    assert len(payload) == (13 * len(ids) + 7) // 8


def test_fixed13_canonical_bytes_and_rejects_incomplete_or_dirty_padding():
    assert pack_tokens_13bit([1, 2]) == bytes.fromhex("01400000")
    with pytest.raises(ValueError):
        unpack_tokens_13bit(bytes.fromhex("01400080"), 2)
    with pytest.raises(ValueError):
        unpack_tokens_13bit(bytes.fromhex("014000"), 2)
    with pytest.raises(ValueError):
        pack_tokens_13bit([8192])
    with pytest.raises(TypeError):
        pack_tokens_13bit([1.5])
    assert unpack_tokens_13bit(pack_tokens_13bit(map(operator.index, [0, 5])), 2) == [0, 5]


def test_rate_counts_written_padding_and_only_real_source_frames():
    from hla_tok.streaming.fixed13 import payload_ledger

    payloads = [pack_tokens_13bit([5, 6, 8191]), pack_tokens_13bit([7])]
    size = sum(len(p) for p in payloads)
    rate = payload_ledger(size, 4, 3, 1080, 1920)
    assert rate["padding_bits"] == 4
    assert rate["total_bytes"] == 7
    assert rate["token_per_frame"] == pytest.approx(4 / 3)
    assert rate["actual_bpp"] == pytest.approx(56 / (3 * 1080 * 1920))
