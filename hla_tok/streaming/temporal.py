"""Causal HLT v1 encoder and a bytes-only, incremental decoder."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from hla_tok.streaming.temporal_wire import (
    HEADER_BYTES,
    NO_DIGEST,
    NO_REFERENCE,
    RECORD_BYTES,
    FrameHeader,
    Mode,
    Record,
    open_header,
    open_record,
    record_bytes,
)
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import Profile
from hla_tok.temporal.quantization import quantize_symbols, reconstruct
from hla_tok.temporal.sparse import Blocks, decode_rows, unpack_block
from hla_tok.temporal.state import BaseReferenceState, DisplayReconstructionState, commit, predict, use_intra

NATIVE_COEFFICIENT_SHAPE = (LAYOUT.token_count, LAYOUT.components)


class BudgetTooSmall(ValueError):
    """The budget cannot carry the complete base for this frame."""


class ReferenceMismatch(ValueError):
    """A P frame could not be attached to the committed base history."""


@dataclass(frozen=True)
class EncodedFrame:
    frame_id: int
    raw: bytes
    base_bytes: int
    base: BaseReferenceState
    clipped_symbols: int
    transmitted_symbols: int
    prediction: np.ndarray | int


def _check_sender_reference(frame_id: int, reference: BaseReferenceState | None, profile: Profile) -> None:
    if reference is not None and (reference.profile_binding != profile.binding or reference.frame_id >= frame_id):
        raise ReferenceMismatch("encoder reference frame or Profile differs")


def _encoding_header(frame: int, reference: BaseReferenceState | None, profile: Profile,
                     target: np.ndarray, force_intra: bool) -> FrameHeader:
    _check_sender_reference(frame, reference, profile)
    intra = use_intra(frame, reference, profile, target, force_intra)
    if intra:
        return FrameHeader(frame, NO_REFERENCE, profile.profile_id, profile.base_records, profile.binding, NO_DIGEST)
    if reference is None or reference.frame_id + 1 != frame:
        raise ReferenceMismatch("P frame must refer to the immediately preceding base")
    return FrameHeader(frame, reference.frame_id, profile.profile_id, profile.base_records,
                       profile.binding, reference.digest)


class _PrefixWriter:
    def __init__(self, profile: Profile, header: FrameHeader, budget: int, payload_budget: int | None):
        self.profile = profile
        self.parts = [header.serialize()]
        self.size = len(self.parts[0])
        self.payload_size = 0
        self.clipped = 0
        self.sent = 0
        self.budget = budget
        self.payload_budget = payload_budget

    def _fit(self, lengths: np.ndarray) -> int:
        payload = np.cumsum(lengths)
        used = payload + RECORD_BYTES * np.arange(1, len(lengths) + 1)
        fit = int(np.searchsorted(used, self.budget - self.size, side="right"))
        if self.payload_budget is not None:
            fit = min(fit, int(np.searchsorted(payload, self.payload_budget - self.payload_size, side="right")))
        if fit:
            self.size += int(used[fit - 1])
            self.payload_size += int(payload[fit - 1])
        return fit

    def write(self, first: int, stop: int, target: np.ndarray, prediction: np.ndarray | int,
              steps: np.ndarray, base: bool) -> np.ndarray:
        profile = self.profile
        first_slot, last_slot = profile.starts[first], profile.starts[stop]
        width = (last_slot - first_slot) // (stop - first)
        span = slice(first_slot, last_slot)
        pred = prediction[span] if isinstance(prediction, np.ndarray) else prediction
        codes, clipped = quantize_symbols(target[profile.scan[span]], pred, steps[span], profile.bits)
        blocks = Blocks(codes.reshape(-1, width), profile.bits, profile.sparse)
        fit = self._fit(blocks.lengths)
        if base and fit < stop - first:
            raise BudgetTooSmall("byte budget cannot carry the complete base")
        selected = fit * width
        self.parts.extend(record_bytes(first + offset, profile.starts[first + offset], width, base, mode, payload)
                          for offset, (mode, payload) in enumerate(blocks.payloads(fit)))
        self.sent = first_slot + selected
        self.clipped += int(np.count_nonzero(clipped[:selected]))
        return codes

    def ranges(self, first: int, stop: int) -> Iterator[tuple[int, int]]:
        width = self.profile.block_size
        starts = self.profile.starts
        ragged = starts[stop] - starts[stop - 1] < width if stop > first else False
        whole = stop - int(ragged)
        for index in range(first, whole, 96):
            yield index, min(index + 96, whole)
        if ragged:
            yield whole, stop


def _prefix(target: np.ndarray, profile: Profile, header: FrameHeader, prediction: np.ndarray | int,
            budget: int, payload_budget: int | None) -> tuple[bytes, int, np.ndarray, int, int]:
    writer = _PrefixWriter(profile, header, budget, payload_budget)
    steps = profile.steps_i if header.intra else profile.steps_p
    first_enhancement = profile.base_records
    base = np.concatenate([writer.write(first, stop, target, prediction, steps, True)
                           for first, stop in writer.ranges(0, first_enhancement)])
    base_bytes = writer.size
    for first, stop in writer.ranges(first_enhancement, profile.records):
        if writer.size + RECORD_BYTES > budget:
            break
        writer.write(first, stop, target, 0, profile.steps_i, False)
        if writer.sent < profile.starts[stop]:
            break
    return b"".join(writer.parts), base_bytes, base, writer.clipped, writer.sent


def encode_frame(target: np.ndarray, frame_id: int, reference: BaseReferenceState | None, profile: Profile,
                 budget: int, *, payload_budget: int | None = None, force_intra: bool = False) -> EncodedFrame:
    """Encode each emitted Record at its final position, with a budget-independent base."""
    if target.dtype != np.dtype("<i2") or target.shape != NATIVE_COEFFICIENT_SHAPE:
        raise ValueError("expected signed int16 native lattice coordinates")
    if budget < HEADER_BYTES or (payload_budget is not None and payload_budget < 0):
        raise BudgetTooSmall("byte budget cannot carry an HLT header")
    flat = target.reshape(-1)
    header = _encoding_header(frame_id, reference, profile, flat, force_intra)
    prediction = predict(profile, reference, header.intra)
    raw, base_bytes, codes, clipped, sent = _prefix(flat, profile, header, prediction, budget, payload_budget)
    state = commit(profile, header, reference, prediction, codes)
    return EncodedFrame(frame_id, raw, base_bytes, state, clipped, sent, prediction)


def encode_frames(frames: Iterable[tuple[np.ndarray, int]], profile: Profile) -> Iterator[EncodedFrame]:
    reference = None
    for frame_id, (target, budget) in enumerate(frames):
        frame = encode_frame(target, frame_id, reference, profile, budget)
        reference = frame.base
        yield frame


@dataclass
class WireLedger:
    header_bytes: int = HEADER_BYTES
    crc_bytes: int = 4
    record_header_bytes: int = 0
    base_payload_bytes: int = 0
    enhancement_payload_bytes: int = 0
    bitmap_bytes: int = 0
    padding_bits: int = 0

    def add_batch(self, *, count: int, payload: int, base_payload: int, bitmap: int, padding: int) -> None:
        self.record_header_bytes += count * (RECORD_BYTES - 4)
        self.crc_bytes += count * 4
        self.base_payload_bytes += base_payload
        self.enhancement_payload_bytes += payload - base_payload
        self.bitmap_bytes += bitmap
        self.padding_bits += padding

    def totals(self) -> dict[str, int]:
        payload = self.base_payload_bytes + self.enhancement_payload_bytes
        return vars(self) | {"header_bytes": self.header_bytes - 4, "payload_bytes": payload,
                             "motion_payload_bytes": 0, "padding_bytes": 0,
                             "total_bytes": self.header_bytes - 4 + self.crc_bytes
                             + self.record_header_bytes + payload}


class _Frame:
    def __init__(self, profile: Profile, header: FrameHeader, prior: BaseReferenceState | None):
        self.profile = profile
        self.header = header
        self.prior = prior
        self.prediction = predict(profile, prior, header.intra)
        self.codes = np.empty(len(profile.scan), dtype=np.int8)
        self.seen: set[int] = set()
        self.prefix = 0
        self.ledger = WireLedger()
        self.base: BaseReferenceState | None = None

    def _verify_range(self, record: Record) -> int:
        index = record.index
        if not 0 <= index < self.profile.records or index in self.seen:
            raise ValueError("duplicate or out-of-range HLT Record")
        start, stop = self.profile.starts[index:index + 2]
        if (record.start, record.count, record.base) != (start, stop - start, index < self.profile.base_records):
            raise ValueError("overlapping or invalid HLT scan range")
        if not self.profile.mode_allowed(record.mode):
            raise ValueError("coding mode conflicts with the bound Profile")
        return stop

    def _advance(self) -> None:
        while self.prefix in self.seen:
            self.prefix += 1

    def add_record(self, record: Record) -> None:
        stop = self._verify_range(record)
        symbols, padding = unpack_block(record.mode, record.payload, record.count)
        self.codes[record.start:stop] = symbols
        self.seen.add(record.index)
        self.ledger.add_batch(count=1, payload=len(record.payload),
                              base_payload=len(record.payload) if record.base else 0,
                              bitmap=(record.count + 7) // 8 if record.mode == Mode.BITMAP4 else 0,
                              padding=padding)
        self._advance()

    def add_stream(self, raw: memoryview) -> None:
        records: dict[tuple[Mode, int], list[Record]] = defaultdict(list)
        offset = HEADER_BYTES
        while offset < len(raw):
            record, offset = open_record(raw, offset)
            self._verify_range(record)
            self.seen.add(record.index)
            records[(record.mode, record.count)].append(record)
        for (mode, count), batch in records.items():
            self._decode_batch(mode, count, batch)
        self._advance()

    def _decode_batch(self, mode: Mode, count: int, batch: list[Record]) -> None:
        payloads = [record.payload for record in batch]
        values, padding = decode_rows(mode, payloads, count)
        starts = np.array([record.start for record in batch], dtype=np.int32)
        self.codes[starts[:, None] + np.arange(count, dtype=np.int32)] = values
        self.ledger.add_batch(count=len(batch), payload=sum(map(len, payloads)),
                              base_payload=sum(len(record.payload) for record in batch if record.base),
                              bitmap=len(batch) * ((count + 7) // 8) if mode == Mode.BITMAP4 else 0,
                              padding=padding)

    def commit_base(self) -> BaseReferenceState:
        if self.prefix < self.profile.base_records:
            raise ValueError("incomplete HLT base prefix")
        if self.base is None:
            codes = self.codes[:self.profile.base_components]
            self.base = commit(self.profile, self.header, self.prior, self.prediction, codes)
        return self.base

    def render(self, records: int | None) -> DisplayReconstructionState:
        base = self.commit_base()
        count = self.prefix if records is None else records
        if not self.profile.base_records <= count <= self.prefix:
            raise ValueError("render requires a received, contiguous legal Record prefix")
        width = self.profile.base_components
        stop = self.profile.end(count)
        enhancement = slice(width, stop)
        display = np.zeros(NATIVE_COEFFICIENT_SHAPE, dtype="<i2")
        display.reshape(-1)[self.profile.scan[:width]] = base.coordinates[self.profile.scan_base_order]
        display.reshape(-1)[self.profile.scan[enhancement]] = reconstruct(
            0, self.codes[enhancement], self.profile.steps_i[enhancement])
        support = base.support.copy()
        support[self.profile.scan_tokens[enhancement]] = True
        return DisplayReconstructionState(display, support, count)


class TemporalReceiver:
    """The receiver owns no analysis network, source frames, or encoder cache."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self.reference: BaseReferenceState | None = None
        self._active: _Frame | None = None
        self._last_seen = -1

    @property
    def prefix_records(self) -> int:
        return self._active.prefix if self._active is not None else 0

    @property
    def waiting_for_intra(self) -> bool:
        return self.reference is None

    def _require_frame(self, frame_id: int) -> _Frame:
        if self._active is None or self._active.header.frame_id != frame_id:
            raise ValueError("frame is outside the open HLT assembly")
        return self._active

    def _prior(self, header: FrameHeader) -> BaseReferenceState | None:
        if header.intra:
            return None
        prior = self.reference
        if prior is None or header.frame_id != header.reference_id + 1 or header.reference_id != prior.frame_id:
            self.reference = None
            raise ReferenceMismatch("the named previous base is unavailable")
        if header.reference_digest != prior.digest:
            self.reference = None
            raise ReferenceMismatch("reference state digest differs")
        return prior

    def begin_frame(self, raw_header: bytes | memoryview) -> int:
        header = open_header(raw_header)
        if (header.profile_id, header.binding) != (self.profile.profile_id, self.profile.binding):
            raise ValueError("HLT profile or asset binding differs")
        if header.base_records != self.profile.base_records:
            raise ValueError("HLT base boundary differs from the bound scan")
        if self._active is not None or header.frame_id <= self._last_seen:
            raise ValueError("HLT frame must be newer and the previous frame finalized")
        self._active = _Frame(self.profile, header, self._prior(header))
        self._last_seen = header.frame_id
        return header.frame_id

    def receive_record(self, raw_record: bytes) -> None:
        if self._active is None:
            raise ValueError("an HLT frame header is required")
        record, stop = open_record(raw_record)
        if stop != len(raw_record):
            raise ValueError("expected a single complete HLT Record")
        self._active.add_record(record)

    def receive_stream(self, raw: bytes) -> int:
        view = memoryview(raw)
        old_last = self._last_seen
        frame_id = self.begin_frame(view[:HEADER_BYTES])
        try:
            self._require_frame(frame_id).add_stream(view)
        except ValueError:
            self._active = None
            self._last_seen = old_last
            raise
        return frame_id

    def commit_base(self, frame_id: int) -> BaseReferenceState:
        self.reference = self._require_frame(frame_id).commit_base()
        return self.reference

    def render_prefix(self, frame_id: int, record_count: int | None = None) -> DisplayReconstructionState:
        return self._require_frame(frame_id).render(record_count)

    def wire_ledger(self, frame_id: int) -> dict[str, int]:
        return self._require_frame(frame_id).ledger.totals()

    def finalize_frame(self, frame_id: int) -> bool:
        active = self._require_frame(frame_id)
        complete = active.prefix >= self.profile.base_records
        self.reference = active.commit_base() if complete else None
        self._active = None
        return complete

    def reset_reference(self) -> None:
        self.reference = None
        self._active = None
        self._last_seen = -1
