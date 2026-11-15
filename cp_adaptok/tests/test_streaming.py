import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from cp_adaptok.streaming.bitstream import StreamError
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.transport import receive_record, send_record


@torch.no_grad()
def test_actual_localhost_progressive_records(codec):
    encoded = CodecEncoder(codec).encode_frame(torch.rand(1, 3, 24, 32))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def receive():
            with torch.no_grad(), listener.accept()[0] as connection:
                connection.settimeout(10)
                decoder = CodecDecoder(codec)
                decoder.start_frame(receive_record(connection))
                decoder.append_packets([receive_record(connection)])
                base = decoder.decode_base().clone()
                fingerprint = decoder.get_state().fingerprint()
                for _ in range(3):
                    decoder.append_packets([receive_record(connection)])
                full = decoder.decode_prefix(32, use_cache=True)
                assert decoder.get_state().fingerprint() == fingerprint
                return base, full

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(receive)
            with socket.create_connection(listener.getsockname(), timeout=10) as connection:
                total = send_record(connection, encoded.header_bytes)
                total += sum(send_record(connection, p) for p in encoded.packets)
            base, full = future.result(timeout=15)
    assert base.shape == full.shape == (1, 3, 24, 32)
    assert total == encoded.bytes_at(32)


@torch.no_grad()
def test_wrong_model_reference_corrupt_duplicate_and_enhancement_loss(codec):
    encoder, decoder = CodecEncoder(codec), CodecDecoder(codec)
    first = encoder.encode_frame(torch.rand(1, 3, 24, 32))
    with pytest.raises(StreamError, match="model_id"):
        decoder.start_frame(replace(first.header, model_id="ff" * 16).to_bytes())
    decoder.start_frame(first.header_bytes)
    decoder.append_packets(first.packets[:1])
    decoder.decode_base()
    with pytest.raises(StreamError, match="contiguous"):
        decoder.append_packets(first.packets[:1])
    # Deliberately discard every enhancement packet, then advance normally.
    second = encoder.encode_frame(torch.rand(1, 3, 24, 32))
    with pytest.raises(StreamError, match="fingerprint"):
        decoder.start_frame(replace(second.header, reference_fingerprint="ee" * 16).to_bytes())
    decoder.start_frame(second.header_bytes)
    decoder.append_packets(second.packets[:1])
    decoder.decode_base()
    assert decoder.get_state().fingerprint() == encoder.get_state().fingerprint()


@torch.no_grad()
def test_online_sender_selects_actual_immutable_packets(codec):
    encoder = CodecEncoder(codec)
    encoded = encoder.encode_frame(torch.rand(1, 3, 16, 24))
    result = encoder.select_prefix(encoded, 100000, 0.2, mode="rate-only")
    point, packets = result
    assert packets == encoded.packets[: point.tokens // codec.config.packet_tokens]
    assert encoded.bytes_at(point.tokens) <= 2500
    assert encoder.select_prefix(encoded, 8, 0.001) is None
