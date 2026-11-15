"""Causal prefix scheduling over reliable, self-describing socket records.

The receiver knows neither the source pixels nor the sender's future budgets.
Display controls live outside the immutable CPAT/CPTK codec records.
"""

import hashlib
import math
import struct
import time
from dataclasses import dataclass

import torch

from .bitstream import StreamError, _checked_body, _crc_record
from .packetizer import validate_budgets
from .receiver import CodecDecoder
from .transport import PacedRecordWriter, receive_record, send_record

DISPLAY = struct.Struct(">4sBQH")
ACK = struct.Struct(">4sBQH16s")
END_SESSION = _crc_record(b"CPEN\x01")
END_ACK = _crc_record(b"CPED\x01")
DISPLAY_EXCHANGE_BYTES = DISPLAY.size + ACK.size + 16  # CRC32 and framing for both records
DISPLAY_RESERVE_MULTIPLIER = 1.25


def _parse_control(wire, layout, magic):
    if len(wire) != layout.size + 4:
        raise StreamError("invalid session control length")
    fields = layout.unpack(_checked_body(wire))
    if fields[:2] != (magic, 1):
        raise StreamError("unsupported session control/version")
    return fields[2:]


@dataclass(frozen=True)
class DisplayRequest:
    frame_id: int
    tokens: int

    def __post_init__(self):
        if type(self.frame_id) is not int or type(self.tokens) is not int:
            raise StreamError("display identity/budget must be integers")
        if not 0 <= self.frame_id < 2**64 or not 0 < self.tokens <= 8192:
            raise StreamError("invalid display identity/budget")

    def to_bytes(self):
        return _crc_record(DISPLAY.pack(b"CPDS", 1, self.frame_id, self.tokens))

    @classmethod
    def from_bytes(cls, wire):
        return cls(*_parse_control(wire, DISPLAY, b"CPDS"))


@dataclass(frozen=True)
class DisplayAck:
    request: DisplayRequest
    state_fingerprint: str

    def __post_init__(self):
        try:
            fingerprint = bytes.fromhex(self.state_fingerprint)
        except (ValueError, TypeError) as error:
            raise StreamError("invalid ACK reference fingerprint") from error
        if len(fingerprint) != 16:
            raise StreamError("ACK reference fingerprint must contain 128 bits")
        object.__setattr__(self, "state_fingerprint", fingerprint.hex())

    def to_bytes(self):
        return _crc_record(
            ACK.pack(
                b"CPAK",
                1,
                self.request.frame_id,
                self.request.tokens,
                bytes.fromhex(self.state_fingerprint),
            )
        )

    @classmethod
    def from_bytes(cls, wire):
        frame, tokens, fingerprint = _parse_control(wire, ACK, b"CPAK")
        return cls(DisplayRequest(frame, tokens), fingerprint.hex())


@dataclass(frozen=True)
class AllocationConfig:
    mode: str = "rate-only"
    playback_delay_s: float = 0.25
    decode_reserve_s: float = 0.025
    buffer_s: float = 0
    fixed_k: int | None = None
    enhancement_cap: int | None = None

    def __post_init__(self):
        values = (self.playback_delay_s, self.decode_reserve_s, self.buffer_s)
        if not all(math.isfinite(v) for v in values) or self.playback_delay_s <= 0 or min(values[1:]) < 0:
            raise ValueError("invalid playback/decode/buffer timing")
        if self.mode not in ("fixed", "rate-only", "scorer"):
            raise ValueError("unknown causal allocation mode")
        if self.mode == "fixed" and self.fixed_k is None:
            raise ValueError("fixed allocation requires fixed_k")


class SocketReceiver:
    """Single independent state owner; display callback completes before its ACK."""

    def __init__(self, connection, model, display=None, ack_delay_s=0):
        if not math.isfinite(ack_delay_s) or ack_delay_s < 0:
            raise ValueError("invalid ACK delay")
        self.connection, self.display, self.ack_delay_s = connection, display, ack_delay_s
        self.decoder = CodecDecoder(model)
        self.events = []
        self.received_bytes = self.ack_bytes = self.displayed_tokens = 0
        self.packet_hash = hashlib.sha256()

    @torch.no_grad()
    def run(self):
        handlers = {b"CPAT": self._start, b"CPTK": self._append, b"CPDS": self._display}
        while True:
            record = receive_record(self.connection)
            self.received_bytes += len(record) + 4
            if record == END_SESSION:
                return self._finish()
            handler = handlers.get(record[:4])
            if handler is None:
                raise StreamError("unknown session record")
            handler(record)

    def _start(self, record):
        self._require_displayed_prefix()
        self.decoder.start_frame(record)
        self.displayed_tokens = 0
        self.packet_hash = hashlib.sha256()

    def _append(self, record):
        self.decoder.append_packets([record])
        self.packet_hash.update(record)

    def _display(self, record):
        request = DisplayRequest.from_bytes(record)
        stream = self.decoder.stream
        if stream is None or request.frame_id != stream.header.frame_id:
            raise StreamError("display request does not identify the current frame")
        if request.tokens != len(stream.token_ids) or request.tokens <= self.displayed_tokens:
            raise StreamError("display request must advance to the received contiguous prefix")
        frame = self.decoder.decode_prefix(request.tokens, use_cache=True)
        if self.display is not None:
            self.display(request.frame_id, request.tokens, frame)
        fingerprint = self.decoder.get_state().fingerprint()
        self.displayed_tokens = request.tokens
        self.events.append(
            {
                "frame": request.frame_id,
                "tokens": request.tokens,
                "display_time": time.perf_counter(),
                "state_fingerprint": fingerprint,
                "packet_sha256": self.packet_hash.hexdigest(),
                "reconstructions": self.decoder.reconstructions,
                **self.decoder.last_timing_ms,
            }
        )
        if self.ack_delay_s:
            time.sleep(self.ack_delay_s)
        self.ack_bytes += send_record(self.connection, DisplayAck(request, fingerprint).to_bytes())

    def _require_displayed_prefix(self):
        if self.decoder.context is not None and not self.decoder.context.committed:
            raise StreamError("reference prefix missing; reliable recovery or reset required")
        if self.decoder.stream is not None and len(self.decoder.stream.token_ids) != self.displayed_tokens:
            raise StreamError("received prefix has not been displayed")

    def _finish(self):
        self._require_displayed_prefix()
        self.ack_bytes += send_record(self.connection, END_ACK)
        return {
            "events": self.events,
            "received_bytes": self.received_bytes,
            "ack_bytes": self.ack_bytes,
            "base_commits": self.decoder.base_commits,
        }


class SocketSender:
    def __init__(self, connection, encoder, budgets, allocation=None):
        validate_budgets(budgets, encoder.model.config)
        if allocation is not None:
            extras = [k for k in (allocation.fixed_k, allocation.enhancement_cap) if k is not None]
            for k in extras:
                validate_budgets([k], encoder.model.config)
            if allocation.fixed_k is not None and allocation.fixed_k not in budgets:
                raise ValueError("fixed_k must be a scored budget")
        self.connection, self.encoder = connection, encoder
        self.budgets, self.allocation = tuple(budgets), allocation
        self.writer = PacedRecordWriter(connection)
        self.ack_bytes = 0
        self.observed_display_ack_s = 0

    @torch.no_grad()
    def send_frame(self, rgb, bandwidth_bps=None, release_time=None):
        if self.allocation is not None and bandwidth_bps is None:
            raise ValueError("online allocation requires the current bandwidth observation")
        self.writer.set_bandwidth(bandwidth_bps)
        release_time = time.perf_counter() if release_time is None else release_time
        if not math.isfinite(release_time):
            raise ValueError("invalid frame release time")
        encoded = self.encoder.encode_frame(rgb, budgets=self.budgets)
        self.writer.send(encoded.header_bytes)
        if self.allocation is None:
            return self._send_all_displays(encoded, release_time)
        return self._send_adaptive(encoded, release_time)

    def _send_packets(self, encoded, first, last, finish_by=None):
        c = self.encoder.model.config
        delivered = first
        for packet in encoded.packets[first // c.packet_tokens : last // c.packet_tokens]:
            if not self.writer.send(packet, finish_by=finish_by):
                break
            delivered += c.packet_tokens
        return delivered

    def _display_ack(self, encoded, k):
        request = DisplayRequest(encoded.header.frame_id, k)
        self.writer.send(request.to_bytes())
        request_sent = time.perf_counter()
        wire = receive_record(self.connection)
        self.ack_bytes += 4 + len(wire)
        expected = DisplayAck(request, self.encoder.get_state().fingerprint())
        if DisplayAck.from_bytes(wire) != expected:
            raise StreamError("display ACK identity/reference mismatch; stop and resynchronize")
        received = time.perf_counter()
        self.observed_display_ack_s = max(self.observed_display_ack_s, received - request_sent)
        return received

    def _send_all_displays(self, encoded, release_time):
        delivered = 0
        for k in self.budgets:
            delivered = self._send_packets(encoded, delivered, k)
            self._display_ack(encoded, delivered)
        return self._frame_report(encoded, delivered, delivered, release_time)

    def _send_adaptive(self, encoded, release_time):
        c, policy = self.encoder.model.config, self.allocation
        deadline = release_time + policy.playback_delay_s
        delivered = self._send_packets(encoded, 0, c.k_ref)
        base_ack = self._display_ack(encoded, delivered)
        decode_reserve = max(
            policy.decode_reserve_s, DISPLAY_RESERVE_MULTIPLIER * self.observed_display_ack_s
        )
        selected = self.encoder.select_prefix(
            encoded,
            self.writer.bandwidth_bps,
            max(0, deadline - time.perf_counter() - decode_reserve),
            buffer_s=policy.buffer_s,
            mode=policy.mode,
            fixed_k=policy.fixed_k,
            delivered_tokens=delivered,
            control_bytes=DISPLAY_EXCHANGE_BYTES,
        )
        target = c.k_ref if selected is None else selected[0].tokens
        limit = min(target, policy.enhancement_cap or target)
        reserve = decode_reserve + DISPLAY_EXCHANGE_BYTES * 8 / self.writer.bandwidth_bps
        delivered = self._send_packets(encoded, delivered, limit, finish_by=deadline - reserve)
        if delivered > c.k_ref:
            self._display_ack(encoded, delivered)
        report = self._frame_report(encoded, target, delivered, release_time)
        report.update(
            {
                "deadline": deadline,
                "base_ack_time": base_ack,
                "base_wait_required": base_ack > deadline,
                "final_ack_before_deadline": time.perf_counter() <= deadline,
                "bandwidth_bps": self.writer.bandwidth_bps,
                "effective_decode_reserve_s": decode_reserve,
                "observed_display_ack_s": self.observed_display_ack_s,
                "selected_predicted_mse": encoded.predicted_mse.get(target),
                "delivered_predicted_mse": encoded.predicted_mse.get(delivered),
                "deadline_or_cap_trimmed_packets": (target - delivered) // c.packet_tokens,
            }
        )
        return report

    def _frame_report(self, encoded, target, delivered, release_time):
        return {
            "frame": encoded.header.frame_id,
            "release_time": release_time,
            "selected_tokens": target,
            "delivered_tokens": delivered,
            "codec_and_framing_bytes": encoded.bytes_at(delivered),
            "state_fingerprint": self.encoder.get_state().fingerprint(),
            "packet_sha256": hashlib.sha256(b"".join(encoded.prefix_packets(delivered))).hexdigest(),
            "full_packet_sha256": hashlib.sha256(b"".join(encoded.packets)).hexdigest(),
            "omitted_enhancement_packets": (encoded.header.n_max - delivered) // encoded.header.packet_tokens,
            "encoding_ms": encoded.timing_ms,
        }

    def finish(self):
        self.writer.send(END_SESSION)
        wire = receive_record(self.connection)
        self.ack_bytes += len(wire) + 4
        if wire != END_ACK:
            raise StreamError("invalid session completion ACK")
