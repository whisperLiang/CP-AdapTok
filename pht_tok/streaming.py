"""Causal prefix selection and an explicit packet/transport deadline model."""

import math
import random
from bisect import bisect_right
from dataclasses import dataclass

MAX_PACKET_PAYLOAD = 1200
PACKET_OVERHEAD = 40  # IPv4 + UDP + RTP base header.


def network_sizes(prefix_sizes: tuple[int, ...]) -> tuple[int, ...]:
    previous = overhead = 0
    values = []
    for current in prefix_sizes:
        if current <= previous:
            raise ValueError("prefix sizes must increase")
        overhead += math.ceil((current - previous) / MAX_PACKET_PAYLOAD) * PACKET_OVERHEAD
        values.append(current + overhead)
        previous = current
    return tuple(values)


def select_prefix(prefix_sizes: tuple[int, ...], estimated_goodput: float, playback_deadline: float,
                  encoding_delay: float, decoding_delay: float, safety_margin: float, *,
                  minimum: int = 1) -> int:
    """Times are seconds relative to the decision, goodput is network bits/s."""
    available = max(0.0, playback_deadline - encoding_delay - decoding_delay - safety_margin)
    budget = available * estimated_goodput / 8
    prefix = bisect_right(network_sizes(prefix_sizes), budget)
    return prefix if prefix >= minimum else 0


@dataclass(frozen=True, slots=True)
class Observation:
    at: float
    bits_per_second: float


def goodput_at(observations: tuple[Observation, ...], now: float, initial: float, alpha: float = 0.45) -> float:
    goodput = initial
    for item in sorted(observations, key=lambda record: record.at):
        if item.at > now:
            break
        goodput = (1 - alpha) * goodput + alpha * item.bits_per_second
    return goodput


def _valid_rate(time: float, rate: float) -> bool:
    return math.isfinite(time) and math.isfinite(rate) and rate > 0


@dataclass(frozen=True, slots=True)
class RateTrace:
    knots: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "knots", tuple((float(t), float(rate)) for t, rate in self.knots))
        if not self.knots or self.knots[0][0] != 0:
            raise ValueError("trace must start at time zero")
        if not all(_valid_rate(time, rate) for time, rate in self.knots):
            raise ValueError("trace needs finite, positive rates")
        if any(a[0] >= b[0] for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError("trace times must increase")

    def finish(self, start: float, byte_count: int) -> float:
        bits = byte_count * 8.0
        index = bisect_right(self.knots, (start, math.inf)) - 1
        while index + 1 < len(self.knots):
            next_time = self.knots[index + 1][0]
            bits_per_second = self.knots[index][1] * 1e6
            capacity = (next_time - start) * bits_per_second
            if bits <= capacity:
                return start + bits / bits_per_second
            bits -= capacity
            start = next_time
            index += 1
        return start + bits / (self.knots[index][1] * 1e6)


@dataclass(frozen=True, slots=True)
class Packet:
    wire_bytes: int
    arrival: float | None


@dataclass(frozen=True, slots=True)
class Delivery:
    arrivals: tuple[float | None, ...]
    packets: tuple[Packet, ...]
    observation: Observation

    @property
    def bytes_sent(self) -> int:
        return sum(packet.wire_bytes for packet in self.packets)

    @property
    def bytes_received(self) -> int:
        return self.received_by(math.inf)

    def received_by(self, deadline: float) -> int:
        return sum(packet.wire_bytes for packet in self.packets
                   if packet.arrival is not None and packet.arrival <= deadline)

    def stage_by(self, deadline: float, decode_seconds: float) -> int:
        return sum(at is not None and at + decode_seconds <= deadline for at in self.arrivals)


class Transport:
    def __init__(self, trace: RateTrace, seed: int, rtt_seconds: float, *,
                 jitter_seconds: float = 0, packet_loss: float = 0):
        self.trace = trace
        self.rng = random.Random(seed)
        self.rtt = rtt_seconds
        self.jitter = jitter_seconds
        self.loss = packet_loss
        self.free = 0.0

    def send(self, ready: float, boundaries: tuple[int, ...]) -> Delivery:
        previous = 0
        arrivals: list[float | None] = []
        packets = []
        start = self.free = max(ready, self.free)
        for boundary in boundaries:
            record = []
            while previous < boundary:
                payload = min(MAX_PACKET_PAYLOAD, boundary - previous)
                previous += payload
                record.append(self._send_packet(payload + PACKET_OVERHEAD))
            packets.extend(record)
            arrivals.append(self._prefix_arrival(record, arrivals))
        received = sum(p.wire_bytes for p in packets if p.arrival is not None)
        observation = Observation(self.free + self.rtt + self.jitter,
                                  8 * received / max(self.free - start, 1e-9))
        return Delivery(tuple(arrivals), tuple(packets), observation)

    def _send_packet(self, size: int) -> Packet:
        self.free = self.trace.finish(self.free, size)
        if self.rng.random() < self.loss:
            return Packet(size, None)
        delay = max(0.0, self.rtt / 2 + self.rng.uniform(-self.jitter, self.jitter))
        return Packet(size, self.free + delay)

    @staticmethod
    def _prefix_arrival(packets: list[Packet], arrivals: list[float | None]) -> float | None:
        prior = arrivals[-1] if arrivals else 0.0
        times = [p.arrival for p in packets if p.arrival is not None]
        return max(prior, *times) if prior is not None and len(times) == len(packets) else None
