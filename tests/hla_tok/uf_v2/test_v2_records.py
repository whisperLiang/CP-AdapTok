import numpy as np
import pytest
import torch

from hla_tok.uf.anchor import IntegerCanvas, RecordCodec, Segmentation
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.wire import FrameParser, PrefixFamily
from hla_tok.uf_v2.records import BatchRecords


@pytest.mark.parametrize("shape", [(3, 8, 12), (5, 9, 13), (8, 1, 1)])
@pytest.mark.parametrize("ranked", [False, True])
def test_batched_wire_is_identical(shape, ranked):
    rng = np.random.default_rng(17)
    values = rng.choice(np.array([0, 0, 0, -127, 127, -8, -9, 1, 3], np.int8), (2, 1, *shape))
    coarse, fine = (torch.from_numpy(item) for item in values)
    rule = Segmentation(4, 2, ranked=ranked)
    legacy = RecordCodec((0.8, 0.5), rule)
    fast = BatchRecords((0.8, 0.5), rule)
    base = BaseFrame(FrameInfo(0, 1080, 1920, 0, True, 1, False), b"base")
    expected = PrefixFamily.build(base, b"model001", legacy.records(coarse, fine), 1100)
    actual = fast.family(base, b"model001", coarse, fine, 1100)
    assert actual == expected

    parser = FrameParser(b"model001")
    reference = IntegerCanvas(legacy, shape)
    canvas = fast.canvas(shape, True)
    for a, b in zip((0, *actual.ends[:-1]), actual.ends, strict=True):
        packet = actual.raw[a:b]
        parser.feed(packet[:-1])
        assert list(parser.available()) == []
        parser.feed(packet[-1:])
        [record] = parser.available()
        if a:
            canvas.apply_batch([record])
            reference.apply(record)
            np.testing.assert_array_equal(canvas.symbols, reference.symbols)
            assert canvas.symbol_count == reference.symbol_count
    parser.finish()
