import random

import numpy as np
import pytest
import torch

from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.wire import Enhancement, WireError, pack_enhancement
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
        first = pack_enhancement(Enhancement(0, 1, 4, b"1234"))
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
    invalid = bytearray(family.raw[:family.base_end])
    invalid[4] ^= 1
    receiver = make()
    receiver.feed(invalid)
    with pytest.raises(WireError):
        receiver.decode_available()
    assert not calls
