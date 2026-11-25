"""In-process experiment endpoints with distinct sender and receiver histories."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from torch import Tensor

from hla_tok.streaming.temporal import EncodedFrame, ReferenceMismatch, TemporalReceiver, encode_frame
from hla_tok.streaming.temporal_wire import HEADER_BYTES, NO_REFERENCE, open_header, open_record
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import Profile
from hla_tok.temporal.state import BaseReferenceState, DisplayReconstructionState


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.square(values, dtype=np.float32).mean()))


@dataclass
class Delivered:
    encoded: EncodedFrame
    display: DisplayReconstructionState | None
    receiver: BaseReferenceState | None
    ledger: dict
    actual_bytes: int
    received_bytes: int
    frame_type: str
    event: str
    encoding_ms: float
    reception_ms: float


class Link:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.sender_reference: BaseReferenceState | None = None
        self.receiver = TemporalReceiver(profile)

    def send(self, target: np.ndarray, frame: int, cap: int, *, payload_cap: int | None = None,
             channel: str = "full") -> Delivered:
        start = perf_counter()
        encoded = encode_frame(target, frame, self.sender_reference, self.profile, cap, payload_budget=payload_cap)
        encoding_ms = 1000 * (perf_counter() - start)
        self.sender_reference = encoded.base
        if len(encoded.raw) > cap:
            raise RuntimeError("serialized HLT bytes exceeded their per-frame ceiling")
        sent = encoded.raw[:encoded.base_bytes] if channel == "base-only" else encoded.raw
        received = sent[:encoded.base_bytes] if channel == "lose-enhancement" else sent
        intra = open_header(encoded.raw).reference_id == NO_REFERENCE
        start = perf_counter()
        display, state, ledger, received_size, event = self._receive(received, frame, channel, intra)
        if state is not None and state.digest != encoded.base.digest:
            raise RuntimeError(f"independent reference mismatch on frame {frame}")
        return Delivered(encoded, display, state, ledger, len(sent), received_size,
                         "I" if intra else "P", event, encoding_ms, 1000 * (perf_counter() - start))

    def _receive(self, raw: bytes, frame: int, channel: str, intra: bool) -> tuple:
        was_waiting = self.receiver.waiting_for_intra
        try:
            if channel == "lose-base":
                return self._lose_base(raw, frame)
            self.receiver.receive_stream(raw)
        except ReferenceMismatch:
            if not was_waiting:
                raise
            return None, None, {}, len(raw), "WAIT_I"
        ledger = self.receiver.wire_ledger(frame)
        if ledger["total_bytes"] != len(raw):
            raise RuntimeError("record byte ledger differs from physical receive length")
        display = self.receiver.render_prefix(frame)
        state = self.receiver.commit_base(frame)
        self.receiver.finalize_frame(frame)
        event = {"base-only": "base_only", "lose-enhancement": "enhancement_discarded"}.get(channel, "received")
        if intra:
            event = "initial_I" if frame == 0 else ("I_recovery" if was_waiting else "I_refresh")
        return display, state, ledger, len(raw), event

    def _lose_base(self, raw: bytes, frame: int) -> tuple:
        self.receiver.begin_frame(raw[:HEADER_BYTES])
        _, missing_stop = open_record(raw, HEADER_BYTES)
        offset = missing_stop
        while offset < len(raw):
            _, stop = open_record(raw, offset)
            self.receiver.receive_record(raw[offset:stop])
            offset = stop
        self.receiver.finalize_frame(frame)
        return None, None, {}, len(raw) - (missing_stop - HEADER_BYTES), "base_record_lost"


def prediction_coordinates(delivered: Delivered, profile: Profile) -> np.ndarray:
    dense = np.zeros((LAYOUT.token_count, LAYOUT.components), dtype=np.int16)
    dense.reshape(-1)[profile.scan[:profile.base_components]] = delivered.encoded.prediction
    return dense


def temporal_error(original: Tensor, pixels: Tensor, previous: tuple[Tensor, Tensor] | None) -> float | None:
    if previous is None:
        return None
    old_source, old_pixels = previous
    return float(((pixels.float() - old_pixels.float()) - (original - old_source)).square().mean())
