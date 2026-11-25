import random

import numpy as np
import pytest
import torch

from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.wire import FrameParser, WireError, pack_base
from hla_tok.uf_v2.records import BatchReceiver, BatchRecords, Display


@pytest.fixture
def stream():
    bank = BatchRecords((0.8, .5), Segmentation(4, 2))
    shape = (4, 9, 13)
    values = np.random.default_rng(7).integers(-3, 3, (2, *shape), dtype=np.int8)
    base = BaseFrame(FrameInfo(0, 1080, 1920, 0, True, 1, False), b"base")
    tag = b"testwire"
    family = bank.pack_symbols(base, tag, values, 10000)
    calls = []

    def decode(frame):
        calls.append(frame)
        return Display(torch.zeros((1, *shape)), bank.canvas(shape, True), torch.ones(shape), lambda y: y)

    return family, lambda: BatchReceiver(tag, decode), calls


def test_fragments_decode_once_and_preserve_base(stream):
    family, make, calls = stream
    full, split = make(), make()
    full.feed(family.raw)
    assert full.decode_available() == len(family.ends)
    full.finish()

    rng = random.Random(13)
    offset = records = 0
    while offset < len(family.raw):
        end = min(offset + rng.randrange(1, 100), len(family.raw))
        split.feed(family.raw[offset:end])
        records += split.decode_available()
        assert split.decode_available() == 0
        offset = end
    split.finish()
    assert records == len(family.ends)
    assert len(calls) == 2
    assert split.symbol_count == full.symbol_count
    assert torch.equal(split.render(), full.render())
    assert torch.equal(split.base, torch.zeros_like(full.base))


@pytest.mark.parametrize("fault", ["crc", "duplicate", "dependency", "length"])
def test_bad_record_does_not_allow_render(stream, fault):
    family, make, calls = stream
    receiver = make()
    receiver.feed(family.raw[:family.base_end])
    assert receiver.decode_available() == 1
    first = bytearray(family.raw[family.base_end:family.ends[1]])
    if fault == "crc":
        first[-1] ^= 1
    elif fault == "duplicate":
        first *= 2
    elif fault == "dependency":
        candidates = [family.raw[start:stop] for start, stop in zip(family.ends, family.ends[1:])]
        first = next(packet for packet in candidates if packet[4] == 1)
    else:
        first[6:8] = b"\x00\x00"
    receiver.feed(first)
    with pytest.raises(WireError):
        receiver.decode_available()
    with pytest.raises(WireError):
        receiver.render()
    assert len(calls) == 1


def test_incomplete_base_and_model_tag(stream):
    family, make, calls = stream
    receiver = make()
    receiver.feed(family.raw[:family.base_end - 1])
    assert receiver.decode_available() == 0
    with pytest.raises(WireError):
        receiver.finish()
    original = FrameParser(b"testwire")
    original.feed(family.raw[:family.base_end])
    [base] = original.available()
    receiver = make()
    receiver.feed(pack_base(base, b"wrongfam"))
    with pytest.raises(WireError):
        receiver.decode_available()
    assert not calls


def test_exact_receiver_cap_is_accepted(monkeypatch, stream):
    import hla_tok.uf_v2.records as module

    family, make, _ = stream
    monkeypatch.setattr(module, "MAX_FRAME_BYTES", len(family.raw))
    receiver = make()
    receiver.feed(family.raw)
    assert receiver.decode_available() == len(family.ends)
    receiver.finish()
    with pytest.raises(WireError, match="length"):
        receiver.feed(b"x")


def test_finish_refuses_truncated_enhancement_and_zero_path_does_not_run_head(stream):
    family, make, _ = stream
    receiver = make()
    receiver.feed(family.raw[:family.ends[1] - 1])
    assert receiver.decode_available() == 1
    assert receiver.render() is receiver.base
    with pytest.raises(WireError, match="incomplete"):
        receiver.finish()
    receiver.feed(family.raw[family.ends[1] - 1:family.ends[1]])
    assert receiver.decode_available() == 1
    assert receiver.render() is not receiver.base
    receiver.finish()


def test_base_cannot_commit_twice(stream):
    family, make, calls = stream
    receiver = make()
    receiver.feed(family.raw[:family.base_end])
    assert receiver.decode_available() == 1
    with pytest.raises(WireError, match="already committed"):
        receiver.commit_base(calls[0])
    assert len(calls) == 1


def test_parsed_enhancements_use_same_receiver_contract(stream):
    family, make, _ = stream
    parser = FrameParser(b"testwire")
    parser.feed(family.raw)
    base, *enhancements = parser.available()
    receiver = make()
    with pytest.raises(WireError, match="requires a complete base"):
        receiver.apply_enhancement(enhancements[0])
    receiver.commit_base(base)
    for record in enhancements:
        receiver.apply_enhancement(record)
    complete = make()
    complete.feed(family.raw)
    assert complete.decode_available() == len(family.ends)
    assert receiver.symbol_count == complete.symbol_count
    assert receiver.received == complete.received == len(family.raw)
    assert torch.equal(receiver.render(), complete.render())
    with pytest.raises(WireError, match="duplicate"):
        receiver.apply_enhancement(enhancements[0])
    with pytest.raises(WireError, match="rejected"):
        receiver.render()


def test_transport_close_keeps_final_render_and_does_not_allow_append(stream):
    family, make, _ = stream
    receiver = make()
    receiver.feed(family.raw)
    assert receiver.decode_available() == len(family.ends)
    expected = receiver.render()
    receiver.close()
    assert torch.equal(expected, receiver.render())
    with pytest.raises(WireError, match="closed by transport"):
        receiver.feed(b"more")


def test_lost_base_invalidates_predicted_reference_until_independent_i():
    from hla_tok.uf.base import ReferenceOrder

    reference = ReferenceOrder()

    def info(i, intra=False):
        return FrameInfo(i, 1080, 1920, 0, intra, 1, False)

    reference.commit(info(0, True))
    with pytest.raises(ValueError, match="DESYNC"):
        reference.commit(info(2))
    with pytest.raises(ValueError, match="DESYNC"):
        reference.commit(info(1))
    reference.commit(info(3, True))
    reference.commit(info(4))
    assert reference.last.index == 4
