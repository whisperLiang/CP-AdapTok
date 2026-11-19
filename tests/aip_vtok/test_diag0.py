import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aip_vtok.decoder.causal_state import FrameContext
from aip_vtok.diagnostics.diag0 import io
from aip_vtok.diagnostics.diag0.basis import fit_basis, latent_rows
from aip_vtok.diagnostics.diag0.closed import receive_frame
from aip_vtok.diagnostics.diag0.fit import ORDERS, train_latents
from aip_vtok.diagnostics.diag0.io import Catalog, sealed_projector
from aip_vtok.diagnostics.diag0.matched import full_rank_latent
from aip_vtok.diagnostics.diag0.sparse import (
    Symbols,
    allocate_depth,
    byte_ledger,
    decode_fsq,
    depth_bytes,
    encode_fsq,
    pack_depth,
    prefix_slots,
    unpack_depth,
)
from aip_vtok.tokenizer.fsq import FSQ


def test_separate_full_rank_bases_restore_their_channel_means():
    rng = torch.Generator().manual_seed(733)
    intra = torch.randn(1, 128, 12, 24, generator=rng) + 2
    predictive = torch.randn(1, 128, 12, 24, generator=rng) - 3
    bases = [fit_basis([value], 128) for value in (intra, predictive)]
    assert not torch.equal(bases[0].vectors, bases[1].vectors)
    for value, basis in zip((intra, predictive), bases):
        rows = latent_rows(value)
        result = basis.restore(basis.rotate(rows))
        assert result.dtype == torch.float32
        assert (result - rows).abs().max() < 1e-5
        torch.testing.assert_close(basis.restore(torch.zeros_like(rows)), basis.mean.expand_as(rows))


def test_fp32_probe_roundtrip_at_native_latent_magnitudes():
    rng = torch.Generator().manual_seed(3421)
    y = torch.randn(1, 128, 32, 24, generator=rng) * 6 + 2
    basis = fit_basis([y], 128)
    fit = SimpleNamespace(basis=basis, indices=lambda _: torch.arange(128))
    restored = full_rank_latent(fit, y, ORDERS[0])
    assert restored.dtype == torch.float32
    assert (restored - y).abs().max() < 1e-5


@pytest.mark.parametrize("split", ["validation", "test"])
def test_fit_refuses_other_splits_before_requesting_any_pixels(split):
    class Catalog:
        def read(self, *_):
            pytest.fail("a held-out image was requested for fitting")

    catalog = Catalog()
    catalog.split = split
    with pytest.raises(ValueError, match="train"):
        train_latents(catalog, None, None)


def test_validation_catalog_supports_sealed_development_frames_but_train_cannot_read_them(monkeypatch):
    rows = {"car": {"role": "historical-development"}, "sealed": {"role": "test"}}
    monkeypatch.setattr(io, "read_window", lambda row, start, length: (row["role"], start, length))
    assert Catalog("validation", rows, {}).read("car", 0, 2) == ("historical-development", 0, 2)
    with pytest.raises(ValueError, match="declared split"):
        Catalog("train", rows, {}).read("car", 0, 2)
    with pytest.raises(ValueError, match="declared split"):
        Catalog("test", rows, {}).read("sealed", 0, 2)


@pytest.mark.parametrize("cell, expected", [(1, [9, 17, 36]), (2, [8, 16, 36])])
def test_depth_packets_are_nested_prefixes_and_have_a_real_packed_map(cell, expected):
    budgets = (9, 17, 36)
    energy = np.random.default_rng(541).uniform(size=(4, 8, 21))
    maps = allocate_depth(energy, budgets, cell)
    old = np.zeros_like(maps[9])
    for budget, count in zip(budgets, expected):
        depth = maps[budget]
        assert np.all(depth >= old)
        assert set(np.unique(depth)) <= {0, 1, 2, 4, 8, 12}
        packed = pack_depth(depth)
        assert len(packed) == (depth.size * 3 + 7) // 8
        np.testing.assert_array_equal(unpack_depth(packed, depth.shape), depth)
        occupied = prefix_slots(depth, cell)
        assert occupied.size == count <= budget
        for slot in occupied:
            pixel, group = divmod(slot, 21)
            assert all(pixel * 21 + g in occupied for g in range(group))
        old = depth


def test_depth_ties_use_raster_order_and_never_skip_a_prefix():
    energy = np.zeros((2, 2, 21))
    energy[..., 0] = 1
    assert allocate_depth(energy, (2,), 1)[2].tolist() == [[1, 1], [0, 0]]


@pytest.mark.parametrize("shape, raw_bytes", [((68, 120), 3060), ((34, 60), 765)])
def test_ledger_counts_exact_packed_depth_and_labels_entropy_reference(shape, raw_bytes):
    depth = np.zeros(shape, dtype=np.uint8)
    depth.ravel()[0:15] = 4
    packed = pack_depth(depth)
    compressed = zlib.compress(packed, level=9)
    np.testing.assert_array_equal(unpack_depth(zlib.decompress(compressed), shape), depth)
    measured = depth_bytes(packed, shape)
    assert measured["R_depth_raw"] == raw_bytes
    assert measured["R_depth_compressed_ref"] == len(compressed)
    counts = np.zeros((6, 8), dtype=int)
    counts[:, :2] = 250
    version = byte_ledger(510, 500, 1080 * 1920, counts, depth=measured)
    assert version["R_tokens_raw"] == 1020
    assert version["R_tokens_raw_used"] == 1000
    assert version["R_tokens_entropy_lb"] == 375
    assert version["R_total_entropy_ref"] == 375 + len(compressed) + 24
    assert byte_ledger(510, 500, 1080 * 1920, depth=measured)["bpp_entropy_ref"] is None


def test_reserved_depth_symbols_and_truncation_are_rejected():
    with pytest.raises(ValueError, match="reserved"):
        unpack_depth(b"\xe0", (1, 1))
    with pytest.raises(ValueError, match="length"):
        unpack_depth(b"", (1, 1))
    with pytest.raises(ValueError, match="no 3-bit symbol"):
        pack_depth(np.array([[3]]))


def test_fsq_decoder_uses_only_integer_ids(monkeypatch):
    fsq = FSQ()
    values = torch.tensor([[0.93, -0.71, 0.88, -0.39, 0.61, 0.12]])
    scale = torch.ones_like(values)
    called = []
    original = fsq.codes_to_ids

    def encode_codes(codes):
        called.append(codes.clone())
        return original(codes)

    monkeypatch.setattr(fsq, "codes_to_ids", encode_codes)
    ids, saturation = encode_fsq(fsq, values, scale)
    assert len(called) == 1 and ids.dtype == torch.int64 and saturation.shape == values.shape
    assert torch.equal(fsq.ids_to_codes(ids), called[0])
    actual = decode_fsq(fsq, ids, scale)
    assert not torch.allclose(actual, values)
    torch.testing.assert_close(Symbols(ids, None).decode(scale, fsq), actual, rtol=0, atol=0)
    monkeypatch.setattr(fsq, "ids_to_codes", lambda _: torch.zeros_like(values))
    torch.testing.assert_close(decode_fsq(fsq, ids, scale), torch.zeros_like(values), atol=1e-7, rtol=0)


@pytest.mark.parametrize("frame, intra, budget", [(0, True, 2048), (1, False, 510), (32, True, 2048)])
def test_display_suffix_does_not_change_next_reference_state(frame, intra, budget):
    memory = None if intra else torch.full((1, 256, 2, 2), 7.0)
    calls = []

    def receive(k):
        calls.append(k)
        return torch.full((1, 128, 1, 1), float(k))

    def render(latent, context):
        return latent * 2, None

    def projector(latent, intra):
        return latent[:, :1].expand(1, 192 if intra else 256, 2, 2)

    low, low_next = receive_frame(FrameContext(frame, intra, None, memory), receive, render, projector, (2048,))
    high, high_next = receive_frame(FrameContext(frame, intra, None, memory), receive, render, projector, (8192,))
    assert calls == [2048, budget, 8192, budget]
    assert not torch.equal(low[2048][0], high[8192][0])
    assert low_next.fingerprint() == high_next.fingerprint()
    assert torch.all(low_next.latent == budget)


def test_gate0_projector_matches_sealed_sha256():
    if not io.PROJECTOR.is_file():
        pytest.skip("requires the original local Gate 0 checkpoint")
    record = sealed_projector()
    assert record["sha256"] == "3b1bd715427bd177393e9552a99576e614ab7dd1ed307155129cf62896d5eb94"
