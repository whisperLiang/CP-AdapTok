import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hla_tok.streaming.temporal import TemporalReceiver, encode_frame
from hla_tok.streaming.temporal_wire import HEADER_BYTES, NO_DIGEST, open_record
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.native_wire import NativeWire
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE, Profile


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("optional HLT native wire tests require a C compiler")
    root = Path(__file__).resolve().parents[3]
    source = root / "hla_tok/temporal/records.c"
    output = tmp_path_factory.mktemp("hlt-wire") / "records.so"
    subprocess.run([compiler, "-O3", "-std=c11", "-fPIC", "-shared", str(source), "-o", str(output), "-lz"],
                   capture_output=True, check=True)
    return output


def profile(bits, block):
    shape = (LAYOUT.token_count, LAYOUT.components)
    native = np.arange(np.prod(shape)).reshape(shape)
    base = native[:135, 0]
    extra = native[:270, 1:7].flatten()
    steps = np.full(RANK_COMPONENT_SHAPE, 2)
    return Profile(1, np.hstack([base, extra]), len(base), steps, steps, NO_DIGEST, bits=bits, block_size=block)


@pytest.mark.parametrize("bits,blocks", [(4, 64), (4, 128), (4, 256), (8, 256)])
def test_native_backend_matches_bytes_display_and_ledger(library, bits, blocks):
    using = profile(bits, blocks)
    implementation = NativeWire(library, using)
    fallback, fast = TemporalReceiver(using), TemporalReceiver(using, wire=implementation)
    rng = np.random.default_rng(13)
    reference = None
    for frame, budget in enumerate((850, 1500, 400, 1800)):
        source = rng.integers(-32, 34, (LAYOUT.token_count, LAYOUT.components), dtype=np.int16)
        if frame % 2:
            source[source % 8 != 0] = 0
        if frame == 0:
            source *= 0
        a = encode_frame(source, frame, reference, using, budget)
        b = encode_frame(source, frame, reference, using, budget, wire=implementation)
        assert a.raw == b.raw
        assert a.base.digest == b.base.digest and a.clipped_symbols == b.clipped_symbols
        for endpoint in (fallback, fast):
            endpoint.receive_stream(a.raw)
            endpoint.commit_base(frame)
        assert fallback.wire_ledger(frame) == fast.wire_ledger(frame)
        v1, v2 = fallback.render_prefix(frame), fast.render_prefix(frame)
        assert np.array_equal(v1.coordinates, v2.coordinates)
        assert np.array_equal(v1.support, v2.support)
        for endpoint in (fallback, fast):
            endpoint.finalize_frame(frame)
        reference = a.base


def test_native_backend_rejects_corruption_and_misbound_profile(library):
    using = profile(4, 64)
    other = replace(using, sparse=False)
    impl = NativeWire(library, using)
    with pytest.raises(ValueError, match="Profile"):
        TemporalReceiver(other, wire=impl)
    values = np.zeros((LAYOUT.token_count, LAYOUT.components), dtype=np.int16)
    raw = encode_frame(values, 0, None, using, 1500).raw
    first, stop = open_record(raw, HEADER_BYTES)
    packets = [
        (raw[:-1] + bytes([raw[-1] ^ 8]), "CRC"),
        (raw + raw[HEADER_BYTES:stop], "duplicate"),
        (raw[:HEADER_BYTES] + replace(first, start=1).serialize() + raw[stop:], "scan"),
        (raw[:-1], "header|length"),
    ]
    for packet, error in packets:
        receiver = TemporalReceiver(using, wire=impl)
        with pytest.raises(ValueError, match=error):
            receiver.receive_stream(packet)
        assert receiver.reference is None
        assert receiver.receive_stream(raw) == 0
        assert receiver.finalize_frame(0)


def test_native_stream_accepts_late_records_and_checks_buffer_bounds(library):
    using = profile(4, 256)
    impl = NativeWire(library, using)
    source = np.full((LAYOUT.token_count, LAYOUT.components), 2, dtype=np.int16)
    frame = encode_frame(source, 0, None, using, 2000, wire=impl)
    with pytest.raises(ValueError, match="writable"):
        impl.open(memoryview(frame.raw), np.empty(1, dtype=np.int8))
    with pytest.raises(ValueError, match="contiguous"):
        impl.pack(np.empty(0, dtype=np.int8), np.ones(1, dtype=np.uint8),
                  np.ones(1, dtype=np.int32), 1, 64, 0, 0, True)
    with pytest.raises(ValueError, match="dimensions"):
        impl.pack(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.uint8),
                  np.empty(0, dtype=np.int32), 0, 64, 0, 0, True)
    receiver = TemporalReceiver(using, wire=impl)
    receiver.receive_stream(frame.raw[:frame.base_bytes])
    before = receiver.commit_base(0)
    snapshot = before.coordinates.copy()
    stop = frame.base_bytes
    while stop < len(frame.raw):
        _, offset = open_record(frame.raw, stop)
        receiver.receive_record(frame.raw[stop:offset])
        stop = offset
    assert receiver.commit_base(0).digest == frame.base.digest
    assert np.array_equal(snapshot, before.coordinates)
    assert receiver.render_prefix(0).records > using.base_records
