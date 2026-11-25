from dataclasses import replace
from hashlib import sha256
from random import Random

import numpy as np
import pytest

from hla_tok.streaming.temporal import (
    BudgetTooSmall,
    ReferenceMismatch,
    TemporalReceiver,
    encode_frame,
    encode_frames,
)
from hla_tok.streaming.temporal_wire import HEADER_BYTES, NO_REFERENCE, Record, open_header, open_record
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import Profile

SIZE = LAYOUT.token_count * LAYOUT.components


@pytest.fixture
def profile():
    coordinates = np.arange(SIZE, dtype=np.int32).reshape(-1, LAYOUT.components)
    base = coordinates[:270, :1].flatten()
    extra = coordinates[:270, 1:5].flatten()
    return Profile(1, np.concatenate((base, extra)), len(base), np.full((121, 32), 3),
                   np.full((121, 32), 2), sha256(b"test frozen models and assets").digest(),
                   bits=4, block_size=64)


def target(offset=0):
    native = np.zeros((LAYOUT.token_count, LAYOUT.components), dtype=np.int16)
    native[:270, :5] = np.arange(5) * 4 + offset
    return native


def records(raw):
    stop = HEADER_BYTES
    while stop < len(raw):
        _, end = open_record(raw, stop)
        yield raw[stop:end]
        stop = end


def received(receiver, frame):
    receiver.receive_stream(frame.raw)
    state = receiver.commit_base(frame.frame_id)
    assert state.digest == frame.base.digest
    display = receiver.render_prefix(frame.frame_id)
    receiver.finalize_frame(frame.frame_id)
    return display


def test_intra_and_predictive_closed_loop(profile):
    receiver = TemporalReceiver(profile)
    first = encode_frame(target(), 0, None, profile, 650)
    assert open_header(first.raw).reference_id == NO_REFERENCE
    assert received(receiver, first).coordinates[0, :5].tolist() == [0, 3, 9, 12, 15]
    second = encode_frame(target(11), 1, first.base, profile, 650)
    assert open_header(second.raw).reference_id == 0
    display = received(receiver, second)
    assert display.coordinates[0, 0] == 12
    assert second.base.coordinates[0] == 12
    assert display.support[:270].all()
    assert not display.support[270:].any()


def test_independent_budgets_share_wire_and_integer_prefix(profile):
    first = encode_frame(target(1), 0, None, profile, 1700)
    large = encode_frame(target(8), 1, first.base, profile, 1700)
    small = encode_frame(target(8), 1, first.base, profile, large.base_bytes + 70)
    assert len(small.raw) < len(large.raw)
    assert large.raw.startswith(small.raw)
    large_rx = TemporalReceiver(profile)
    small_rx = TemporalReceiver(profile)
    for receiver in (large_rx, small_rx):
        received(receiver, first)
    small_rx.receive_stream(small.raw)
    large_rx.receive_stream(large.raw)
    small_display = small_rx.render_prefix(1)
    last = len(tuple(records(small.raw)))
    large_at_small = large_rx.render_prefix(1, last)
    assert np.array_equal(small_display.coordinates, large_at_small.coordinates)
    assert np.array_equal(small_display.support, large_at_small.support)
    with pytest.raises(BudgetTooSmall):
        encode_frame(target(8), 1, first.base, profile, large.base_bytes - 1)


def test_incremental_reordering_duplicate_overlap_and_crc(profile):
    frame = encode_frame(target(6), 0, None, profile, 1700)
    chunks = list(records(frame.raw))
    receiver = TemporalReceiver(profile)
    receiver.begin_frame(frame.raw[:HEADER_BYTES])
    with pytest.raises(ValueError, match="incomplete"):
        receiver.commit_base(0)
    receiver.receive_record(chunks[1])
    assert receiver.prefix_records == 0
    with pytest.raises(ValueError, match="duplicate"):
        receiver.receive_record(chunks[1])
    valid, _ = open_record(chunks[0])
    with pytest.raises(ValueError, match="scan"):
        receiver.receive_record(replace(valid, start=1).serialize())
    with pytest.raises(ValueError, match="CRC"):
        receiver.receive_record(chunks[0][:-1] + bytes([chunks[0][-1] ^ 1]))
    receiver.receive_record(chunks[0])
    assert receiver.prefix_records == 2
    for raw in chunks[2:profile.base_records]:
        receiver.receive_record(raw)
    base = receiver.commit_base(0)
    snapshot = base.coordinates.tobytes()
    before = receiver.render_prefix(0)
    for raw in chunks[profile.base_records:]:
        receiver.receive_record(raw)
    after = receiver.render_prefix(0)
    assert not np.shares_memory(before.coordinates, after.coordinates)
    assert not np.shares_memory(after.coordinates, base.coordinates)
    assert not np.shares_memory(after.support, base.support)
    with pytest.raises(ValueError):
        base.coordinates[0] = 13
    assert base.coordinates.tobytes() == snapshot
    assert base.digest == frame.base.digest
    assert np.any(before.coordinates != after.coordinates)
    direct = TemporalReceiver(profile)
    assert np.array_equal(received(direct, frame).coordinates, after.coordinates)


def test_invalid_binding_frame_id_and_history_digest(profile):
    first = encode_frame(target(9), 10, None, profile, 800)
    second = encode_frame(target(11), 11, first.base, profile, 800)
    receiver = TemporalReceiver(profile)
    received(receiver, first)
    header = open_header(second.raw)
    for altered in (replace(header, reference_id=9), replace(header, reference_digest=bytes(32))):
        with pytest.raises(ReferenceMismatch):
            receiver.begin_frame(altered.serialize())
        receiver.reset_reference()
        received(receiver, first)
    with pytest.raises(ValueError, match="binding"):
        receiver.begin_frame(replace(header, binding=bytes(32)).serialize())
    with pytest.raises(ValueError, match="boundary"):
        receiver.begin_frame(replace(header, base_records=header.base_records + 1).serialize())


def test_incomplete_base_enters_wait_and_recovers_at_intra(profile):
    first = encode_frame(target(), 0, None, profile, 1600)
    second = encode_frame(target(9), 1, first.base, profile, 1600)
    third = encode_frame(target(6), 2, second.base, profile, 1600)
    refresh = encode_frame(target(8), 3, third.base, profile, 1600, force_intra=True)
    receiver = TemporalReceiver(profile)
    received(receiver, first)
    receiver.begin_frame(second.raw[:HEADER_BYTES])
    for chunk in list(records(second.raw))[1:]:
        receiver.receive_record(chunk)
    with pytest.raises(ValueError, match="incomplete"):
        receiver.commit_base(1)
    assert not receiver.finalize_frame(1)
    assert receiver.waiting_for_intra
    with pytest.raises(ReferenceMismatch):
        receiver.receive_stream(third.raw)
    assert received(receiver, refresh).support[:270].all()
    assert not receiver.waiting_for_intra


def test_stream_does_not_fetch_or_depend_on_a_future_frame(profile):
    def guarded():
        yield target(8), 750
        raise AssertionError("read future frame before yielding present frame")

    frame = next(encode_frames(guarded(), profile))
    for future in (target(2), target(23)):
        candidate = next(encode_frames(iter(((target(8), 750), (future, 1700))), profile))
        assert frame.raw == candidate.raw
        assert frame.base.digest == candidate.base.digest


@pytest.mark.parametrize("predictor", ["copy", "linear"])
def test_independent_history_over_70_frames_is_enhancement_independent(profile, predictor):
    profile = profile.with_predictor(predictor, refresh=16)
    stable = TemporalReceiver(profile)
    changing = TemporalReceiver(profile)
    full_reference = short_reference = None
    rng = Random(45)
    for frame_id in range(70):
        rgb_tokens = target((frame_id // 7) % 9)
        full = encode_frame(rgb_tokens, frame_id, full_reference, profile, 1600)
        short = encode_frame(rgb_tokens, frame_id, short_reference, profile, rng.choice((320, 400, 650, 1600)))
        assert full.raw.startswith(short.raw)
        if frame_id % 6 < 3:
            short = replace(short, raw=short.raw[:short.base_bytes])
        received(stable, full)
        received(changing, short)
        assert stable.reference is not None and changing.reference is not None
        assert stable.reference.digest == changing.reference.digest
        assert full.base.digest == short.base.digest
        full_reference, short_reference = full.base, short.base
        assert open_header(full.raw).intra == (frame_id % 16 == 0)


def test_legal_record_truncations_render_and_survive(profile):
    frame = encode_frame(target(5), 0, None, profile, 1700)
    pieces = list(records(frame.raw))
    for count in range(profile.base_records, len(pieces) + 1):
        rx = TemporalReceiver(profile)
        rx.receive_stream(frame.raw[:HEADER_BYTES] + b"".join(pieces[:count]))
        assert rx.prefix_records == count
        assert rx.commit_base(0).digest == frame.base.digest
    rx = TemporalReceiver(profile)
    with pytest.raises(ValueError, match="length"):
        rx.receive_stream(frame.raw[:-1])
    assert rx.reference is None


def test_decoder_checks_profile_coding_mode(profile):
    raw = encode_frame(target(6), 0, None, profile, 1600).raw
    record, _ = open_record(raw, HEADER_BYTES)
    rx = TemporalReceiver(profile)
    rx.begin_frame(raw[:HEADER_BYTES])
    with pytest.raises(ValueError, match="mode"):
        rx.receive_record(Record(record.index, record.start, record.count, record.base, 1,
                                 bytes(record.count)).serialize())


def test_malformed_stream_leaves_no_committable_record_or_index(profile):
    frame = encode_frame(target(6), 0, None, profile, 1600)
    broken = frame.raw[:-1] + bytes([frame.raw[-1] ^ 1])
    receiver = TemporalReceiver(profile)
    with pytest.raises(ValueError, match="CRC"):
        receiver.receive_stream(broken)
    with pytest.raises(ValueError, match="open"):
        receiver.commit_base(0)
    assert received(receiver, frame).support[:270].all()


def test_scene_uses_current_and_committed_base_and_recovers_without_header(profile):
    profile = profile.with_predictor("copy", refresh=64, scene_threshold=0.85)
    first = encode_frame(target(8), 0, None, profile, 1600)
    second = encode_frame(target(8), 1, first.base, profile, 1600)
    third = encode_frame(target(8), 2, second.base, profile, 1600)
    cut = encode_frame(target(-139), 3, third.base, profile, 1600)
    receiver = TemporalReceiver(profile)
    received(receiver, first)
    assert not open_header(second.raw).intra
    with pytest.raises(ReferenceMismatch):
        receiver.receive_stream(third.raw)
    assert receiver.waiting_for_intra
    assert open_header(cut.raw).intra
    received(receiver, cut)


def test_record_sizes_bitmap_crc_and_payload_limit(profile):
    controlled = np.zeros_like(target())
    controlled[5:270:60, 1] = 4
    first = encode_frame(controlled, 0, None, profile, 700, payload_budget=392)
    receiver = TemporalReceiver(profile)
    receiver.receive_stream(first.raw)
    ledger = receiver.wire_ledger(0)
    assert ledger["bitmap_bytes"] > 0
    assert ledger["motion_payload_bytes"] == 0
    assert ledger["total_bytes"] == len(first.raw)
    assert ledger["payload_bytes"] <= 392
    assert sum(ledger[key] for key in ("header_bytes", "record_header_bytes", "crc_bytes",
                                       "base_payload_bytes", "enhancement_payload_bytes")) == len(first.raw)
    assert all(open_record(piece)[1] == len(piece) for piece in records(first.raw))


@pytest.mark.gpu
def test_cpu_and_cuda_int16_lattice_sources_give_identical_state_and_wire(profile):
    import torch

    before = [target(i) for i in (6, 9, 4)]
    gpu = [torch.tensor(item, device="cuda").cpu().numpy() for item in before]
    cpu_packets = list(encode_frames(zip(before, [1600] * len(before), strict=True), profile))
    gpu_packets = list(encode_frames(zip(gpu, [1600] * len(gpu), strict=True), profile))
    assert [frame.raw for frame in cpu_packets] == [frame.raw for frame in gpu_packets]
    assert [frame.base.digest for frame in cpu_packets] == [frame.base.digest for frame in gpu_packets]
