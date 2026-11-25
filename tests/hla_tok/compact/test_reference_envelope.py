import pytest

from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit
from scripts.hla_tok.compact.fixed_reference import HEADER_BYTES, FixedWindow, open_window


def test_reference_frames_use_fixed13_and_count_header():
    # 4 temporal groups and 3 spatial tiles, with one short final window.
    native = FixedWindow(start=48, valid=2, groups=4, tiles=3, count=5)
    indices = [8191 - i for i in range(native.tokens)]
    packet = native.packet(pack_tokens_13bit(indices))
    observed, payload = open_window(packet)
    assert observed == native
    assert unpack_tokens_13bit(payload, observed.tokens) == indices
    assert len(packet) == len(payload) + HEADER_BYTES == 98 + 24
    with pytest.raises(ValueError, match="integrity"):
        open_window(packet[:-1] + bytes([packet[-1] ^ 1]))
