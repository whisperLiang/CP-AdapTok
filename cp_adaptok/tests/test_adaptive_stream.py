import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.transport import PacedRecordWriter, receive_record


@torch.no_grad()
def test_after_base_allocation_charges_only_unsent_bytes(codec):
    encoder = CodecEncoder(codec)
    encoded = encoder.encode_frame(torch.rand(1, 3, 16, 24), budgets=[8, 16, 32])
    overhead = 60
    remaining = encoded.bytes_at(16) - encoded.bytes_at(8) + overhead
    result = encoder.select_prefix(
        encoded,
        bandwidth_bps=8 * remaining,
        deadline_s=1,
        mode="rate-only",
        delivered_tokens=8,
        control_bytes=overhead,
    )
    point, packets = result
    assert point.tokens == 16
    assert point.bytes == remaining
    assert packets == encoded.prefix_packets(16)
    assert encoder.select_prefix(encoded, 8, 0.001, delivered_tokens=8, control_bytes=overhead) is None
    with pytest.raises(ValueError):
        encoder.select_prefix(encoded, 8000, 1, delivered_tokens=9)
    with pytest.raises(ValueError):
        encoder.select_prefix(encoded, 8000, 1, control_bytes=-1)


def test_paced_writer_sends_real_records_and_never_starts_expired_enhancement():
    sender, receiver = socket.socketpair()
    with sender, receiver:
        receiver.settimeout(2)
        writer = PacedRecordWriter(sender, bandwidth_bps=16000)
        payload = b"token-packet" * 10
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(receive_record, receiver)
            start = time.perf_counter()
            assert writer.send(payload) == len(payload) + 4
            assert future.result(timeout=2) == payload
            assert time.perf_counter() - start >= (len(payload) + 4) * 8 / 16000
        assert writer.send(b"expired", finish_by=time.perf_counter()) == 0
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            receive_record(receiver)
        assert writer.bytes_sent == len(payload) + 4


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_pacing_rejects_invalid_rate_without_sending(rate):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        with pytest.raises(ValueError):
            PacedRecordWriter(sender, bandwidth_bps=rate)
