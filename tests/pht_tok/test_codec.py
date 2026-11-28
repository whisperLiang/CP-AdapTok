import os
from pathlib import Path

import pytest
import torch

from baselines.dcvc_uf.native import NativeNetworks
from pht_tok.bitstream import Mode
from pht_tok.codec import MissingReference, PHTDecoder, PHTEncoder, assemble
from third_party.DCVC.src.utils.common import set_torch_env

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def peers():
    weights = os.environ.get("PHT_WEIGHTS")
    if not weights or not torch.cuda.is_available():
        pytest.skip("requires native extensions and official checkpoints")
    set_torch_env()
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        tx = PHTEncoder(NativeNetworks(Path(weights), "hts", 0.15), b"test-id!")
        rx = PHTDecoder(NativeNetworks(Path(weights), "hts", 0.15), b"test-id!")
        frames = torch.rand((17, 3, 80, 96), dtype=torch.float16, device="cuda",
                            generator=torch.Generator("cuda").manual_seed(19)) - 0.5
        yield tx, rx, frames


def test_assemble_preserves_input_order_and_pads_only_the_tail(peers):
    _, _, frames = peers
    assembled = assemble(frames[:3]).reshape(8, 3, *frames.shape[-2:])
    assert torch.equal(assembled[:3], frames[:3])
    assert torch.equal(assembled[3:], frames[2:3].expand(5, -1, -1, -1))


def test_mask_and_precision_incremental_decode_refresh_and_reference_loss(peers):
    tx, rx, frames = peers
    tx.key = rx.key = None
    for i, mode in enumerate((Mode.MASK, Mode.PRECISION)):
        start = 4 * i
        anchor = tx.encode_intra(frames[:1], 32, start)
        rx.decode_prefix(anchor.data)
        first = tx.encode_chunk_once(frames[1:9], 32, start + 1, 2, mode=mode)
        direct = [[image.clone() for image in rx.decode_prefix(first.data[:boundary], commit=False).reconstruct()]
                  for boundary in first.prefix_boundaries]
        session = rx.begin_chunk()
        last = 0
        for stage, boundary in enumerate(first.prefix_boundaries, 1):
            session.append_bytes(first.data[last:boundary - 1])
            assert session.decode_new_layers() == stage - 1
            session.append_bytes(first.data[boundary - 1:boundary])
            assert session.decode_new_layers() == stage
            previous = session.entropy_decodes
            assert session.decode_new_layers() == stage
            assert session.entropy_decodes == previous == stage + 1
            if stage == 2:
                reference = rx.adapter.reference.digest()
            if stage >= 2:
                assert reference == rx.adapter.reference.digest() == tx.adapter.reference.digest()
            display = session.reconstruct()
            assert len(display) == 8
            assert all(torch.equal(a, b) for a, b in zip(display, direct[stage - 1], strict=True))
            assert session.reconstruct() is display
            last = boundary
        second = tx.encode_chunk_once(frames[9:15], 48, start + 2, 2, mode=mode, reset=True)
        low = rx.decode_prefix(second.data[:second.prefix_boundaries[0]])
        assert len(low.reconstruct()) == 6
        assert not low.reference_committed
        third = tx.encode_chunk_once(frames[9:17], 16, start + 3, 2, mode=mode)
        with pytest.raises(MissingReference):
            rx.decode_prefix(third.data)


def test_one_encoder_call_and_chunk_causality_with_shared_backing_storage(peers):
    from unittest.mock import patch

    tx, _, frames = peers
    tx.key = None
    tx.encode_intra(frames[:1], 32, 0)
    with patch.object(tx.adapter, "encode_chunk", wraps=tx.adapter.encode_chunk) as analysis:
        original = tx.encode_chunk_once(frames[1:9], 32, 1, 2, commit=False)
    assert analysis.call_count == 1
    changed_future = frames.clone()
    changed_future[9:] = -changed_future[9:]
    future = tx.encode_chunk_once(changed_future[1:9], 32, 1, 2, commit=False)
    assert original.data == future.data
    changed_current = frames.clone()
    changed_current[8] = -changed_current[8]
    current = tx.encode_chunk_once(changed_current[1:9], 32, 1, 2, commit=False)
    assert original.data != current.data
