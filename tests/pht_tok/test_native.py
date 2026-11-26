import os
from pathlib import Path

import pytest
import torch

from hla_tok.hts.native import NativeNetworks, frame_input
from hla_tok.hts.units import NativeSchedule
from pht_tok.official_adapter import HTSAdapter

pytestmark = pytest.mark.gpu


def test_official_cuda_symbols_replay_on_independent_peer():
    weight_dir = os.environ.get("PHT_WEIGHTS")
    if not weight_dir or not torch.cuda.is_available():
        pytest.skip("requires the pinned DCVC CUDA extensions and PHT_WEIGHTS")
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        weights = Path(weight_dir)
        stock = NativeNetworks(weights, "hts", 0.15)
        tx = HTSAdapter(NativeNetworks(weights, "hts", 0.15).video)
        rx = HTSAdapter(NativeNetworks(weights, "hts", 0.15).video)
        rgb = torch.rand((17, 3, 80, 96), device="cuda", generator=torch.Generator("cuda").manual_seed(5))
        units = iter(NativeSchedule(8, reset_interval=8).units(len(rgb)))
        anchor = stock.encode(rgb, next(units), 32)
        initial = stock.decode(anchor)[0].clone()
        tx.initialize(initial)
        rx.initialize(initial)
        stock.encode(rgb, NativeSchedule(8).units(len(rgb)).__next__(), 32)
        stock_encoded = [stock.encode(rgb, unit, 32) for unit in NativeSchedule(8, reset_interval=8).units(
            len(rgb)) if not unit.intra]
        stock.decode(anchor)
        for unit, expected_bytes in zip(units, stock_encoded, strict=True):
            analysis = tx.encode_chunk(frame_input(rgb, unit), 32)
            rx.start_decoding(analysis.hyper, 32)
            for group in range(4):
                rx.apply_group(analysis.residuals, group)
            display = rx.reconstruct(4, 32, reset=unit.reset)
            expected = stock.decode(expected_bytes)
            official_z, official_y, official_indexes, official_skip = stock.video.proxy.pht_symbols(False)
            assert torch.equal(analysis.hyper, official_z)
            assert torch.equal(analysis.residuals, official_y)
            assert torch.equal(analysis.packed.bitwise_and(255).to(torch.uint8), official_indexes)
            assert torch.equal(analysis.keep, official_skip)
            assert all(torch.equal(a, b) for a, b in zip(display, expected, strict=True))
            state = rx.snapshot_reference(unit.reset)
            assert all(torch.equal(a, b) for a, b in zip(
                state.tensors, stock.video.proxy.audit_reference(unit.reset), strict=True))
            for group in range(4):
                tx.apply_group(analysis.residuals, group)
            tx.reconstruct(4, 32, reset=unit.reset, display=False)
            tx.commit_reference(tx.snapshot_reference(unit.reset))
            rx.commit_reference(state)
