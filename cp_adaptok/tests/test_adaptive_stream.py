import hashlib
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from cp_adaptok.cli.stream_demo import localhost_demo
from cp_adaptok.streaming.bitstream import StreamError
from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.session import (
    END_SESSION,
    AllocationConfig,
    DisplayAck,
    DisplayRequest,
    SocketReceiver,
    SocketSender,
)
from cp_adaptok.streaming.transport import PacedRecordWriter, receive_record, send_record


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
    base, _ = encoder.select_prefix(
        encoded, 8, 0.001, delivered_tokens=8, control_bytes=overhead, mode="rate-only"
    )
    assert base.tokens == 8 and base.bytes == 0
    with pytest.raises(ValueError):
        encoder.select_prefix(encoded, 8000, 1, delivered_tokens=9)
    with pytest.raises(ValueError):
        encoder.select_prefix(encoded, 8000, 1, control_bytes=-1)


@torch.no_grad()
def test_scorer_can_keep_displayed_base_when_every_enhancement_is_worse(codec):
    encoder = CodecEncoder(codec)
    encoded = encoder.encode_frame(torch.rand(1, 3, 16, 24), budgets=[8, 16, 32])
    encoded = replace(encoded, predicted_mse={8: 0.1, 16: 0.2, 32: 0.3})
    point, packets = encoder.select_prefix(
        encoded, 1_000_000, 1, mode="scorer", delivered_tokens=8, control_bytes=62
    )
    assert point.tokens == 8 and point.bytes == 0
    assert packets == encoded.prefix_packets(8)


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


def test_display_controls_are_versioned_checked_and_identify_reference():
    request = DisplayRequest(7, 128)
    assert DisplayRequest.from_bytes(request.to_bytes()) == request
    ack = DisplayAck(request, "ab" * 16)
    assert DisplayAck.from_bytes(ack.to_bytes()) == ack
    for wire in (request.to_bytes()[:-1], request.to_bytes() + b"extra"):
        with pytest.raises(StreamError):
            DisplayRequest.from_bytes(wire)
    damaged = bytearray(ack.to_bytes())
    damaged[-1] ^= 1
    with pytest.raises(StreamError, match="CRC"):
        DisplayAck.from_bytes(bytes(damaged))
    with pytest.raises(StreamError):
        DisplayRequest(0, 0)


@torch.no_grad()
def test_online_socket_truncation_preserves_bytes_and_multiframe_state(codec):
    clip = torch.rand(3, 3, 16, 24)
    reference = CodecEncoder(codec)
    expected = [reference.encode_frame(rgb[None]) for rgb in clip]
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        receiver = SocketReceiver(right, codec)
        future = pool.submit(receiver.run)
        sender = SocketSender(
            left,
            CodecEncoder(codec),
            [8, 16, 32],
            AllocationConfig(mode="fixed", fixed_k=32, playback_delay_s=2, enhancement_cap=16),
        )
        rows = [sender.send_frame(rgb[None], bandwidth_bps=1_000_000) for rgb in clip]
        sender.finish()
        report = future.result(timeout=5)
    assert [e["tokens"] for e in report["events"]] == [8, 16] * len(clip)
    assert all(row["selected_tokens"] == 32 and row["delivered_tokens"] == 16 for row in rows)
    for row, encoded in zip(rows, expected):
        assert row["packet_sha256"] == hashlib.sha256(b"".join(encoded.prefix_packets(16))).hexdigest()
        assert row["codec_and_framing_bytes"] == encoded.bytes_at(16)
        events = [e for e in report["events"] if e["frame"] == row["frame"]]
        assert all(e["state_fingerprint"] == row["state_fingerprint"] for e in events)
        assert events[-1]["packet_sha256"] == row["packet_sha256"]
    assert receiver.decoder.get_state().fingerprint() == reference.get_state().fingerprint()
    assert sender.encoder.base_commits == receiver.decoder.base_commits == len(clip)
    assert sender.writer.bytes_sent == report["received_bytes"]
    assert sender.ack_bytes == report["ack_bytes"]


@torch.no_grad()
def test_expired_frames_deliver_base_reliably_before_next_reference(codec):
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(1) as pool:
        left.settimeout(5)
        right.settimeout(5)
        receiver = SocketReceiver(right, codec)
        future = pool.submit(receiver.run)
        sender = SocketSender(left, CodecEncoder(codec), [8, 16, 32], AllocationConfig(playback_delay_s=1e-6))
        rows = [sender.send_frame(torch.rand(1, 3, 16, 24), bandwidth_bps=1_000_000) for _ in range(3)]
        sender.finish()
        report = future.result(timeout=5)
    assert all(r["base_wait_required"] and r["delivered_tokens"] == 8 for r in rows)
    assert [e["tokens"] for e in report["events"]] == [8, 8, 8]
    assert receiver.decoder.get_state().fingerprint() == sender.encoder.get_state().fingerprint()


@torch.no_grad()
def test_session_cannot_report_success_with_missing_reference_packet(codec):
    encoded = CodecEncoder(codec).encode_frame(torch.rand(1, 3, 16, 24))
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(1) as pool:
        right.settimeout(2)
        receiver = SocketReceiver(right, codec)
        future = pool.submit(receiver.run)
        send_record(left, encoded.header_bytes)
        send_record(left, END_SESSION)
        with pytest.raises(StreamError, match="reference"):
            future.result(timeout=3)
    assert receiver.decoder.base_commits == 0


@pytest.mark.parametrize(
    "options", [{"playback_delay_s": 0}, {"decode_reserve_s": -1}, {"mode": "oracle"}, {"mode": "fixed"}]
)
def test_online_policy_rejects_invalid_external_settings(options):
    with pytest.raises(ValueError):
        AllocationConfig(**options)


@torch.no_grad()
def test_cli_demo_uses_real_tcp_current_rates_and_exact_bidirectional_bytes(codec, tmp_path):
    video = tmp_path / "source.mkv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 20, (24, 16))
    assert writer.isOpened()
    for value in (40, 90, 180):
        writer.write(np.full((16, 24, 3), value, dtype=np.uint8))
    writer.release()
    config = {
        "data": {"test": [{"path": str(video), "frames": 3, "fps": 20}]},
        "evaluation": {"budgets": [8, 16, 32]},
    }
    report = localhost_demo(
        codec,
        config,
        True,
        tmp_path / "output",
        allocation=AllocationConfig(mode="fixed", fixed_k=32, playback_delay_s=1, enhancement_cap=16),
        bandwidth_kbps=(2000, 500, 1000),
        save_frames=False,
        ack_delay_s=0.001,
    )
    assert report["mode"] == "localhost_tcp"
    assert report["bandwidth_kbps_observations"] == [2000, 500, 1000]
    assert report["sender_base_commits"] == report["receiver_base_commits"] == 3
    assert report["total_application_bytes"] == (
        report["codec_and_framing_bytes"]
        + report["forward_control_and_framing_bytes"]
        + report["ack_control_and_framing_bytes"]
    )
    assert all(row["delivered_tokens"] == 16 for row in report["sender_frames"])
    assert report["receiver_inputs"] == "model weights and socket records only"
    assert not list((tmp_path / "output").glob("*.png"))
