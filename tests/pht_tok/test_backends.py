import os
from pathlib import Path

import pytest
import torch

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from baselines.dcvc_uf.native import NativeNetworks
from pht_tok.bitstream import HEADER_BYTES, CorruptStream, Mode
from pht_tok.codec import MissingReference, PHTDecoder, PHTEncoder, assemble
from third_party.DCVC.src.utils.common import set_torch_env

pytestmark = pytest.mark.gpu


def coding_packets(network, frames, delay):
    packets = []
    height, width = frames.shape[-2:]
    pad = (-height) % 16, (-width) % 16
    for start in range(0, len(frames), delay):
        if start == 0:
            encoded = network.intra.compress(frames[:1].contiguous(memory_format=torch.channels_last), 32, *pad)
            network.video.add_ref_feature_from_frame(encoded["x_hat"])
            info = FrameInfo(0, height, width, 32, True, encoded["ec_parallel"], False)
            packets.append(BaseFrame(info, encoded["bit_stream"]))
            continue
        index = start - delay + 1
        unit = frames[index:index + delay]
        x = assemble(unit) if delay == 8 else unit.contiguous(memory_format=torch.channels_last)
        reset = index == 1 + delay
        encoded = network.video.compress(x, 32, reset, *pad)
        info = FrameInfo(index, height, width, 32, False, encoded["ec_parallel"], reset)
        packets.append(BaseFrame(info, encoded["bit_stream"]))
    return packets


@pytest.fixture(params=[("ld", Mode.LD_MASK, 1), ("hts", Mode.MASK_FINE, 8)], scope="module")
def peers(request):
    name, mode, delay = request.param
    weights = os.environ.get("PHT_WEIGHTS")
    if not weights or not torch.cuda.is_available():
        pytest.skip("requires the dual CUDA extension and official checkpoints")
    weight_dir = Path(weights)
    set_torch_env()
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        tx = PHTEncoder(NativeNetworks(weight_dir, name, 0.15), b"test-id!", backend=name)
        rx = PHTDecoder(NativeNetworks(weight_dir, name, 0.15), b"test-id!", backend=name)
        stock = NativeNetworks(weight_dir, name, 0.15)
        rng = torch.Generator("cuda").manual_seed(29)
        frames = torch.rand((1 + 4 * delay, 3, 80, 96), dtype=torch.float16, device="cuda", generator=rng) - 0.5
        yield tx, rx, stock, frames, mode, delay


def test_split_full_prefix_recovers_exact_native_symbols_rgb_and_reference(peers):
    tx, rx, stock, frames, mode, delay = peers
    tx.key = rx.key = None
    packets = coding_packets(stock, frames, delay)
    packet = tx.encode_intra(frames[:1], 32, 0)
    anchor = rx.decode_prefix(packet.data).reconstruct()[0]
    assert torch.equal(anchor, stock.decode(packets[0])[0])
    for native in packets[1:]:
        i, reset = native.info.index, native.info.reset
        encoded = tx.encode_chunk_once(frames[i:i + delay], 32, i, 8, mode=mode, reset=reset)
        session = rx.decode_prefix(encoded.data)
        result = session.reconstruct()
        expected = stock.decode(native)
        assert all(torch.equal(a, b) for a, b in zip(result, expected, strict=True))
        z, y, _, _ = stock.video.proxy.pht_symbols(False)
        assert torch.equal(session.residuals(), y)
        assert torch.equal(tx.adapter.proxy.pht_symbols(True)[0], z)
        expected_ref = stock.video.proxy.audit_reference(reset)
        for a, b, c in zip(tx.adapter.reference.tensors, rx.adapter.reference.tensors, expected_ref, strict=True):
            assert torch.equal(a, b)
            assert torch.equal(b, c)


def test_mixed_budgets_and_fragmentation_keep_the_committed_reference(peers):
    tx, rx, _, frames, mode, delay = peers
    tx.key = rx.key = None
    rx.decode_prefix(tx.encode_intra(frames[:1], 32, 0).data)
    for j, budget in enumerate((8, 1, 5, 8)):
        i = 1 + j * delay
        encoded = tx.encode_chunk_once(frames[i:i + delay], 32 + j, i, 1, mode=mode, reset=j == 2)
        boundary = encoded.prefix_boundaries[budget - 1]
        partial = encoded.data[:boundary - 1]
        session = rx.begin_chunk()
        session.append_bytes(partial, offset=0)
        session.append_bytes(partial, offset=0)
        assert session.decode_new_layers() == budget - 1
        session.append_bytes(encoded.data[boundary - 1:boundary], offset=boundary - 1)
        assert session.decode_new_layers() == budget
        assert session.reference_committed
        assert session.entropy_decodes == budget + 1
        # Once committed, further display work leaves the complete reference tensor tuple stable.
        session.reconstruct()
        for a, b in zip(tx.adapter.reference.tensors, rx.adapter.reference.tensors, strict=True):
            assert torch.equal(a, b)
    last = tx.encode_chunk_once(frames[-delay:], 32, len(frames), 2, mode=mode)
    partial = rx.decode_prefix(last.data[:last.prefix_boundaries[0]])
    assert partial.reconstruct() and not partial.reference_committed
    dependent = tx.encode_chunk_once(frames[-delay:], 32, len(frames) + delay, 2, mode=mode)
    with pytest.raises(MissingReference):
        rx.decode_prefix(dependent.data)


def test_backend_mismatch_and_truncated_header_precede_native_decoding(peers):
    tx, rx, _, frames, mode, _ = peers
    tx.key = rx.key = None
    anchor = tx.encode_intra(frames[:1], 32, 0)
    session = rx.decode_prefix(anchor.data[:HEADER_BYTES - 1])
    assert session.stage == 0 and rx.key is None
    with pytest.raises(MissingReference, match="display"):
        session.reconstruct()
    from dataclasses import replace

    from pht_tok.bitstream import Chunk, ChunkHeader, StreamReader
    parser = StreamReader()
    records = parser.append_bytes(anchor.data)
    other = Mode.INTRA if mode is Mode.LD_MASK else Mode.LD_INTRA
    header = replace(ChunkHeader.unpack(anchor.data[:HEADER_BYTES]), mode=other)
    with pytest.raises(CorruptStream, match="backend"):
        rx.decode_prefix(Chunk.pack(header, records).data)
    assert rx.key is None


def test_modified_reference_is_rejected_and_an_i_frame_recovers(peers):
    tx, rx, _, frames, mode, delay = peers
    tx.key = rx.key = None
    anchor = tx.encode_intra(frames[:1], 32, 0)
    rx.decode_prefix(anchor.data)
    packet = tx.encode_chunk_once(frames[1:1 + delay], 32, 1, 1, mode=mode)
    before = rx.key
    rx.adapter.reference.tensors[0].zero_()
    with pytest.raises(ValueError, match="reference tensors"):
        rx.decode_prefix(packet.data)
    assert rx.key == before
    restart = tx.encode_intra(frames[:1], 32, 1 + delay)
    assert rx.decode_prefix(restart.data).reference_committed
    assert tx.key == rx.key


def test_public_api_encodes_and_reconstructs_native_rgb(peers):
    from pht_tok import PHTTok

    sender, _, _, frames, mode, delay = peers
    codec = PHTTok(sender.backend, Path(os.environ["PHT_WEIGHTS"]), layers=mode.layers, reference_layers=1)
    rgb = (frames.float() + 0.5).mul(255).byte().cpu()
    anchor = codec.encode(rgb[:1], qp=32)
    assert codec.decode(anchor.data).frames.shape == (1, 3, *frames.shape[-2:])
    packet = codec.encode(rgb[1:1 + delay], qp=32)
    minimum = packet.prefix(max_bytes=packet.prefix_boundaries[0])
    assert codec.decode(minimum[:6], offset=0).frames is None
    assert codec.decode(minimum[:6], new_unit=False, offset=0).stage == 0
    result = codec.decode(minimum[6:], new_unit=False, offset=6)
    assert result.stage == 1 and result.reference_committed and result.consumed_bytes == len(minimum)
    assert result.frames.shape == (delay, 3, *frames.shape[-2:])
