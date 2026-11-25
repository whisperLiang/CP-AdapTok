from random import Random

import pytest

from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.wire import (
    APP_BASE_OVERHEAD,
    BaseBudgetExceeded,
    Enhancement,
    FrameParser,
    PrefixFamily,
    WireError,
)

TAG = b"test-UF!"
BASE = BaseFrame(FrameInfo(12, 1080, 1920, 8, True, 8, False), b"z-and-y-native-payload")
ENHANCEMENTS = (Enhancement(13, 0, 0, b"first"), Enhancement(7, 0, 1, b"another"),
                Enhancement(13, 1, 2, b"refinement"))


def test_every_byte_budget_preserves_integer_record_and_byte_prefix():
    family = PrefixFamily.build(BASE, TAG, ENHANCEMENTS)
    full = family.raw
    assert family.base_end == APP_BASE_OVERHEAD + len(BASE.payload)
    for budget in range(family.base_end, len(full) + 1):
        raw, tokens = family.prefix(budget)
        parser = FrameParser(TAG)
        parser.feed(raw)
        recovered = list(parser.available())
        parser.finish()
        assert raw == full[:len(raw)]
        assert len(raw) == max(boundary for boundary in family.ends if boundary <= budget)
        assert recovered == [BASE, *ENHANCEMENTS[:tokens]]
    with pytest.raises(BaseBudgetExceeded, match="BASE_BUDGET_EXCEEDED"):
        family.prefix(family.base_end - 1)


@pytest.mark.parametrize("seed", range(4))
def test_random_fragments_emit_records_exactly_once(seed):
    family = PrefixFamily.build(BASE, TAG, ENHANCEMENTS)
    random = Random(seed)
    parser, recovered = FrameParser(TAG), []
    offset = 0
    while offset < len(family.raw):
        stop = offset + random.randint(1, 20)
        parser.feed(family.raw[offset:stop])
        recovered.extend(parser.available())
        assert not list(parser.available())
        offset = stop
    parser.finish()
    assert recovered == [BASE, *ENHANCEMENTS]


def test_half_record_remains_pending_and_finish_rejects_transport_truncation():
    family = PrefixFamily.build(BASE, TAG, ENHANCEMENTS)
    parser = FrameParser(TAG)
    parser.feed(family.raw[:family.base_end + 8])
    assert list(parser.available()) == [BASE]
    with pytest.raises(WireError, match="incomplete"):
        parser.finish()
    parser.feed(family.raw[family.base_end + 8:])
    assert list(parser.available()) == list(ENHANCEMENTS)


@pytest.mark.parametrize("offset", [4, 24, 36, -2])
def test_corruption_rejected_without_emitting_damaged_record(offset):
    raw = bytearray(PrefixFamily.build(BASE, TAG, ENHANCEMENTS).raw)
    raw[offset] ^= 0x80
    parser = FrameParser(TAG)
    parser.feed(bytes(raw))
    with pytest.raises(WireError, match="checksum|model|length"):
        list(parser.available())


def test_wrong_model_and_missing_dependency_and_repeated_record_are_rejected():
    family = PrefixFamily.build(BASE, TAG, ENHANCEMENTS)
    parser = FrameParser(b"badmodel")
    parser.feed(family.raw)
    with pytest.raises(WireError, match="model"):
        list(parser.available())
    for raw, message in ((family.raw[:family.base_end] + family.raw[family.ends[-2]:], "dependency"),
                         (family.raw + family.raw[family.base_end:family.ends[1]], "duplicate")):
        parser = FrameParser(TAG)
        parser.feed(raw)
        with pytest.raises(WireError, match=message):
            list(parser.available())
