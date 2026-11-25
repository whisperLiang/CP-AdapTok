import numpy as np

from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.entropy import PRECISION
from hla_tok.uf.wire import FrameParser
from hla_tok.uf_v2.records import RAW_FLAG, BatchRecords
from hla_tok.uf_v2.static_entropy import ALPHABET, Histogram, StaticAsset, integer_cdf


def test_integer_cdf_sum_nonzero_and_tie_rule():
    zeros = np.zeros((2, ALPHABET), dtype=np.int64)
    zeros[1, 0] = 10000
    cdf = integer_cdf(zeros)
    frequencies = np.diff(cdf)
    assert np.all(frequencies > 0)
    assert np.all(cdf[:, -1] == PRECISION)
    assert frequencies[0, 0] == frequencies[0, -1] + 1
    assert frequencies[1, 0] > 65000


def test_raw_fallback_includes_metadata_and_round_trips(tmp_path):
    counts = Histogram(4)
    template = np.zeros((2, 4, 8, 8), dtype=np.int8)
    for intra in (False, True):
        counts.observe(intra, np.tile(template, (1, 1, 5, 5)))
    fitted = counts.fit()
    import json

    path = tmp_path / "cdf.json"
    path.write_text(json.dumps(fitted.as_json()))
    loaded = StaticAsset.load(path)
    assert loaded.digest == fitted.digest
    source = np.full_like(template, 127)
    source[1] = 0
    rule = Segmentation(8, 4)
    base = BaseFrame(FrameInfo(0, 1080, 1920, 0, True, 1, False), b"base")
    pure = BatchRecords((0.8, 0.5), rule, tables=loaded.tables)
    raw = BatchRecords((0.8, 0.5), rule, tables=loaded.tables, raw=True)
    tagged = b"testwire"
    a, b = (bank.pack_symbols(base, tagged, source, 10000) for bank in (pure, raw))
    assert len(b.raw) < len(a.raw)
    parser = FrameParser(tagged)
    parser.feed(b.raw)
    _, record = parser.available()
    parser.finish()
    assert record.cdf & RAW_FLAG
    canvas = raw.canvas(tuple(source.shape[1:]), True)
    canvas.apply_batch([record])
    np.testing.assert_array_equal(canvas.symbols, source[0].astype(np.int16) * 2)


def test_integer_table_exact_largest_remainder():
    counts = np.array([[1, 3, 5, 11, 8, 2] * 3], dtype=np.int64)
    weight = [int(v) + 1 for v in counts[0]]
    expected = [(PRECISION - ALPHABET) * v // sum(weight) + 1 for v in weight]
    remain = [(PRECISION - ALPHABET) * v % sum(weight) for v in weight]
    order = sorted(range(ALPHABET), key=lambda index: (-remain[index], index))
    for index in order[:PRECISION - sum(expected)]:
        expected[index] += 1
    np.testing.assert_array_equal(np.diff(integer_cdf(counts))[0], expected)
