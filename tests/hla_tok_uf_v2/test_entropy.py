import numpy as np
import pytest

from baselines.hla_tok_uf_v2.entropy import EntropyTables, SegmentCoder, cdf_table


@pytest.mark.integration
def test_segment_matches_official_rans_and_round_trips_signed_escape_symbols():
    pytest.importorskip("hla_uf_rans")
    upstream = pytest.importorskip("MLCodec_extensions_cpp")
    tables = EntropyTables(tuple(cdf_table(scale) for scale in (0.3, 2.1)))
    coder = SegmentCoder(tables)
    symbols = np.array([127, -127, 0, 2, -2, 8, -8, 9, -9, 3] * 61, dtype=np.int8)
    official = upstream.RansEncoder()
    official.set_cdf(np.array(tables.cdf, dtype=np.int32), np.full(2, len(tables.cdf[0]), dtype=np.int32), 1)
    official.reset()
    official.encode_y((symbols.astype(np.int16) << 8) | 1)
    official.flush()
    raw = coder.encode(symbols, 1)
    assert raw == official.get_encoded_stream().tobytes()
    np.testing.assert_array_equal(coder.decode(raw, 1, len(symbols)), symbols)
    with pytest.raises(ValueError):
        coder.decode(raw[:-1], 1, len(symbols))
    with pytest.raises(ValueError):
        coder.decode(raw + b"\0", 1, len(symbols))
    with pytest.raises(ValueError):
        coder.decode(raw, 0, len(symbols))


def test_integer_cdf_contract():
    table = cdf_table(0.1)
    assert table[0] == 0 and table[-1] == 65536
    assert all(start < stop for start, stop in zip(table, table[1:]))
    with pytest.raises(ValueError):
        EntropyTables(((0, 0, 65536),))
