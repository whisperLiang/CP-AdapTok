import numpy as np

from pht_tok.precision import PrecisionTables, permitted_symbols, representative


def test_floor_buckets_cover_all_signed_symbols_and_refine_exactly():
    exact = np.arange(-128, 128, dtype=np.int16)
    previous = None
    for width in (8, 4, 2, 1):
        lo = permitted_symbols(exact, width)
        preview = representative(lo, width)
        assert np.all((lo <= exact) & (lo + width > exact))
        assert np.all((lo <= preview) & (lo + width > preview))
        if previous is not None:
            assert np.all((previous <= lo) & (lo + width <= previous + 2 * width))
        previous = lo
    np.testing.assert_array_equal(preview, exact)


def test_conditional_cdfs_are_strict_and_deterministic_even_in_tail():
    cdf = np.array([[0, 65000, 65200, 65400, 65536]], dtype=np.int32)
    first = PrecisionTables.from_original(cdf, np.array([5], dtype=np.int32))
    other = PrecisionTables.from_original(cdf, np.array([5], dtype=np.int32))
    for a, b in zip(first.levels, other.levels, strict=True):
        np.testing.assert_array_equal(a, b)
        assert np.all(a[:, 0] == 0)
        assert np.all(a[:, -1] == 65536)
        assert np.all(np.diff(a, axis=1) > 0)


def test_native_coders_cover_the_full_signed_alphabet():
    import pytest

    writer_module = pytest.importorskip("MLCodec_extensions_cpp")
    native = pytest.importorskip("pht_entropy_native")
    from pht_tok.entropy import ConditionalEntropy

    cdf = np.array([[0, 45000, 53000, 61000, 65536]], dtype=np.int32)
    lengths = np.array([5], dtype=np.int32)
    raw = np.arange(-128, 128, dtype=np.int8)
    indexes = np.zeros(len(raw), dtype=np.uint8)
    writer = writer_module.RansEncoder()
    writer.set_entropy_coder_parallel(1)
    writer.set_cdf(cdf, lengths, 1)
    writer.reset()
    writer.encode_y(raw.astype(np.int16) * 256)
    writer.flush()
    payload = writer.get_encoded_stream().tobytes()
    reader = native.SignedDecoder(cdf, lengths, True)
    np.testing.assert_array_equal(reader.y(payload, indexes), raw)
    with pytest.raises(ValueError, match="invalid or truncated"):
        reader.y(payload[:-1], indexes)

    entropy = ConditionalEntropy(PrecisionTables.from_original(cdf, lengths))
    previous = None
    for stage, layer in enumerate(entropy.encode(indexes, raw)):
        previous = entropy.decode(layer, indexes, stage, previous)
    np.testing.assert_array_equal(entropy.preview(previous, 3), raw)
