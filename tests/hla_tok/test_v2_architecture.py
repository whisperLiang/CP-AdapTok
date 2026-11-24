import hashlib
from collections import Counter

import pytest
import torch
from torch.nn import functional as F

from hla_tok import BUDGETS, Geometry, HLATok
from hla_tok.evaluation.representation_diagnostics import explicit_attention
from hla_tok.models.anchors import anchor_table, gaussian_bias
from hla_tok.models.local import RANK_CHUNK, CrossBlock
from hla_tok.models.rank_causal import RankCausalBlock, RankHistory, rank_causal_mask
from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def test_anchor_grid_uses_deterministic_order_and_two_balanced_rounds():
    table = anchor_table()
    assert table.shape == (512, 2)
    assert torch.equal(table, anchor_table())
    expected = [(x, y) for y in (0, 2, 5, 7) for x in (0, 2, 5, 7)]
    assert [tuple(p) for p in table[:16].tolist()] == expected
    assert len(set(map(tuple, table[:64].tolist()))) == 64
    assert set(Counter(map(tuple, table[:128].tolist())).values()) == {2}
    # At the first tie, row-major chooses the farthest available position.
    assert table[16].tolist() == [1, 1]


def test_anchor_bias_is_additive_matches_reference_and_stays_finite_in_bf16():
    table = anchor_table()
    bias = gaussian_bias(table)
    assert bias.shape == (512, 64)
    assert bias[0, 0].item() == 0.0
    assert bias[0, 1].item() == pytest.approx(-1 / 8)
    assert bias[64, 1].item() == pytest.approx(-1 / (2 * .8**2))
    torch.manual_seed(13)
    block = CrossBlock()
    context = torch.randn(2, 64, 128)
    queries = torch.randn(2, 32, 128)
    weights = block.prepare(context)
    q = block.project_query(queries)
    explicit, expected = explicit_attention(q, weights, bias[:32])
    assert explicit[0, 0, 0, 0] > explicit[0, 0, 0, 63]
    y = F.scaled_dot_product_attention(q, *weights, attn_mask=bias[:32])
    torch.testing.assert_close(expected, y, rtol=1e-5, atol=1e-6)
    raw = torch.randn(1, 3, 128, 128)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        model = HLATok("R2").eval()
        prefix = model.encode_frame(raw, 33)
        assert torch.isfinite(model.decode_prefix(prefix, Geometry(128, 128))).all()


def test_direct_projection_stops_at_two_spatial_passes_and_decodes_independently():
    model = HLATok("R1").eval()
    rx = HLATok("R1").eval()
    rx.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 128, 256)
    g = Geometry(128, 256)
    with torch.inference_mode():
        encoded = model.encode_training(image, 243, quantize=False)
        assert encoded.values.shape == (1, 243, 32)
        assert model.encode_frame(image).shape == (1, 256)
        with pytest.raises(ValueError):
            model.encode_frame(image, 257)
        for p in rx.encoder.parameters():
            p.fill_(float("nan"))
        for p in rx.query_encoder.parameters():
            p.fill_(float("nan"))
        ids = model.encode_frame(image, 243)
        other = (ids + 1) % 8192
        picture = rx.decode_prefix(ids, g)
        assert picture.shape == image.shape
        assert torch.isfinite(picture).all()
        assert not torch.equal(picture, rx.decode_prefix(other, g))
    assert torch.equal(model.query_encoder.anchor_table, HLATok("R2").query_encoder.anchor_table[:128])


def test_rank_mask_uses_absolute_indices_and_cache_carries_history_and_gradients():
    mask = rank_causal_mask(start=32, length=32)
    assert mask.shape == (32, 64)
    assert mask[0, :33].all() and not mask[0, 33:].any()
    assert mask[-1].all()
    torch.manual_seed(7)
    block = RankCausalBlock(128, 4)
    left = torch.randn(2, 32, 128, requires_grad=True)
    right = torch.randn(2, 32, 128)
    cache = RankHistory()
    first = block(left, cache, 0)
    second = block(right, cache, 32)
    assert cache.key.shape == (2, 4, 64, 32)
    second.square().mean().backward()
    assert left.grad.norm() > 0
    alternate = RankHistory()
    torch.testing.assert_close(first, block(left, alternate, 0), rtol=0, atol=0)
    changed = right.clone()
    changed[:, 13:] = 20 - changed[:, 13:]
    shifted = block(changed, alternate, 32)
    assert torch.equal(second[:, :13], shifted[:, :13])
    assert not torch.equal(second[:, 13:], shifted[:, 13:])


@pytest.mark.parametrize("architecture", ("R2", "R3"))
def test_early_prefix_ignores_within_chunk_and_cross_chunk_future_query_parameter_changes(architecture):
    torch.manual_seed(33)
    model = HLATok(architecture)
    image = torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        before = model.encode_training(image, 79, quantize=False).values
        # The prefix contains ranks 0--38 from both regions and rank 39 from the first.
        model.query_encoder.query[40:].add_(10 * torch.randn_like(model.query_encoder.query[40:]))
        after = model.encode_training(image, 79, quantize=False).values
    assert torch.equal(before, after)


@pytest.mark.parametrize("architecture", ("R2", "R3"))
def test_anchor_buffer_persists_and_gradients_reach_vq_and_encoder(architecture):
    torch.manual_seed(62)
    model = HLATok(architecture)
    copy = HLATok(architecture)
    copy.load_state_dict(model.state_dict(), strict=True)
    table = copy.state_dict()["query_encoder.anchor_table"]
    assert table.shape == (512, 2)
    assert copy.architecture_metadata()["anchor_table_sha256"] == hashlib.sha256(
        table.to(torch.uint8).numpy().tobytes()).hexdigest()
    image = torch.rand(1, 3, 128, 128)
    for quantize in (False, True):
        model.zero_grad(set_to_none=True)
        enc = model.encode_training(image, 49, quantize=quantize)
        output = model.decode_training(enc, 49)
        loss = (output - image).square().mean() + enc.codebook_loss + 0.25 * enc.commitment_loss
        loss.backward()
        for module in (model.encoder, model.query_encoder, model.local_decoder, model.rgb_decoder):
            assert any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
        gradient = model.quantizer.embedding.weight.grad
        assert (gradient is not None and gradient.norm() > 0) if quantize else gradient is None


@pytest.mark.gpu
@pytest.mark.parametrize("architecture", ("R2", "R3"))
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_native_integer_prefixes_across_all_budgets_and_actual_payloads(architecture, amp):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if amp and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    torch.manual_seed(12)
    model = HLATok(architecture).cuda().to(memory_format=torch.channels_last).eval()
    rx = HLATok(architecture).cuda().to(memory_format=torch.channels_last).eval()
    rx.load_state_dict(model.state_dict(), strict=True)
    source = torch.rand(1, 3, 1080, 1920, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        full = model.encode_frame(source)
        ctx = model.prepare_frame(source)
        for k in (*BUDGETS, 135 * (2 * RANK_CHUNK + 1) - 3):
            assert torch.equal(model.extend_prefix(ctx, k), full[:, :k])
            assert torch.equal(model.encode_frame(source, k), full[:, :k])
        k = 4096
        raw = pack_tokens_13bit(full[0].tolist())
        prefix = raw[:k * 13 // 8]
        ids = torch.tensor([unpack_tokens_13bit(prefix, k)], device="cuda")
        assert torch.equal(ids, full[:, :k])
        altered = full.clone()
        altered[:, k:] = (altered[:, k:] + 1) % 8192
        image = rx.decode_prefix(ids, Geometry(1080, 1920))
        assert image.shape == source.shape and torch.isfinite(image).all()
        assert torch.equal(image, rx.decode_prefix(altered[:, :k], Geometry(1080, 1920)))


def test_production_anchored_query_path_uses_bias():
    torch.manual_seed(8)
    model = HLATok("R2").eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        before = model.encode_training(image, 96, quantize=False).values
        model.query_encoder.spatial_bias.zero_()
        after = model.encode_training(image, 96, quantize=False).values
    assert (before - after).abs().max() > 1e-3


def test_r0_r1_r2_r3_share_exact_initial_spatial_and_decoder_parameters():
    from hla_tok.training.v2 import matched_initialization

    variants = [matched_initialization(name, 1) for name in ("R0", "R1", "R2", "R3")]
    models, initial = zip(*variants)
    assert len({record["shared_sha256"] for record in initial}) == 1
    assert len({record["parameter_count"] for record in initial}) == 3
    a, b = models[2].query_encoder, models[3].query_encoder
    assert torch.equal(a.query, b.query)
    assert torch.equal(a.spatial_bias, b.spatial_bias)


def test_v2_checkpoint_requires_matching_architecture_and_anchor_metadata(tmp_path):
    from hla_tok.training.checkpoint import ResumeIdentity, save_checkpoint
    from hla_tok.training.v2 import load_experimental_checkpoint

    model = HLATok("R3")
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    path = tmp_path / "r3.pt"
    identity = ResumeIdentity("v2/pilot/R3/1", "config", "cohort", "git")
    counters = {"architecture": model.architecture_metadata()}
    save_checkpoint(path, identity, model, optimizer, scheduler, torch.Generator(),
                    step=1, data_cursor=1, counters=counters)
    rebuilt = HLATok("R3")
    load_experimental_checkpoint(path, rebuilt, "cohort")
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), rebuilt.state_dict().values()))
    with pytest.raises(ValueError, match="architecture"):
        load_experimental_checkpoint(path, HLATok("R2"), "cohort")
    with pytest.raises(ValueError, match="manifest"):
        load_experimental_checkpoint(path, rebuilt, "wrong-cohort")
    invalid = tmp_path / "wrong_anchors.pt"
    with torch.no_grad():
        model.query_encoder.anchor_table[0] = torch.tensor([7, 7])
    save_checkpoint(invalid, identity, model, optimizer, scheduler, torch.Generator(),
                    step=1, data_cursor=1, counters=counters)
    with pytest.raises(ValueError, match="anchor"):
        load_experimental_checkpoint(invalid, rebuilt, "cohort")


def test_receiver_cannot_inspect_unreceived_backing_storage():
    torch.manual_seed(71)
    rx = HLATok("R2").eval()
    geometry = Geometry(128, 128)
    common = torch.randint(0, 8192, (1, 45))
    view_a = torch.cat((common, torch.zeros(1, 8, dtype=torch.int64)), 1)[:, :45]
    view_b = torch.cat((common, torch.full((1, 8), 8191, dtype=torch.int64)), 1)[:, :45]
    with torch.inference_mode():
        torch.testing.assert_close(rx.decode_prefix(view_a, geometry), rx.decode_prefix(view_b, geometry),
                                   atol=0, rtol=0)
