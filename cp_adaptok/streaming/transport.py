import math
import struct
import time

from .bitstream import MAX_RECORD_BYTES, StreamError


def record_size(data):
    if not 0 < len(data) <= MAX_RECORD_BYTES:
        raise StreamError("invalid transport record size")
    return 4 + len(data)


def send_record(connection, data):
    size = record_size(data)
    connection.sendall(struct.pack(">I", len(data)) + data)
    return size


def receive_exact(connection, count):
    buffer = bytearray()
    while len(buffer) < count:
        part = connection.recv(count - len(buffer))
        if not part:
            raise EOFError("socket closed before complete record")
        buffer.extend(part)
    return bytes(buffer)


def receive_record(connection):
    size = struct.unpack(">I", receive_exact(connection, 4))[0]
    if not 0 < size <= MAX_RECORD_BYTES:
        raise StreamError("transport record exceeds allocation limit")
    return receive_exact(connection, size)


class PacedRecordWriter:
    """Application-level forward-link serialization; TCP remains reliable and ordered.

    Sleeping before sendall prevents a receiver from seeing a simulated record
    before its wire serialization time. There is no unbounded application queue.
    """

    def __init__(self, connection, bandwidth_bps=None):
        self.connection = connection
        self.bytes_sent = 0
        self.set_bandwidth(bandwidth_bps)

    def set_bandwidth(self, bandwidth_bps):
        if bandwidth_bps is not None and (not math.isfinite(bandwidth_bps) or bandwidth_bps <= 0):
            raise ValueError("bandwidth must be finite and positive")
        self.bandwidth_bps = bandwidth_bps

    def send(self, data, finish_by=None):
        size = record_size(data)
        if finish_by is not None and not math.isfinite(finish_by):
            raise ValueError("deadline must be finite")
        duration = 0 if self.bandwidth_bps is None else 8 * size / self.bandwidth_bps
        if finish_by is not None and time.perf_counter() + duration > finish_by:
            return 0
        if duration:
            time.sleep(duration)
        if finish_by is not None and time.perf_counter() > finish_by:
            return 0
        size = send_record(self.connection, data)
        self.bytes_sent += size
        return size
