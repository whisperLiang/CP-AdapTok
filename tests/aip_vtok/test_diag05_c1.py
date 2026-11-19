"""Behavioral checks for the isolated C1 sender, receiver and frozen boundaries."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from aip_vtok.backbones.dcvc_aligned import ChannelAlignment, Geometry
from aip_vtok.c1.artifacts import frozen_inputs, load_frozen, validate_scan_lock
from aip_vtok.c1.calibrate import split_train
from aip_vtok.c1.channels import ChannelFit, native_gradient
from aip_vtok.c1.dense_eval import BASELINE, INDEPENDENT, PAIRED, TRAINED, dense_decision
from aip_vtok.c1.pilot import require_sparse_release
from aip_vtok.c1.recovery_net import RecoveryNet, commit_reference, reference_supervision
from aip_vtok.c1.scan import BUDGETS, DENSE, SHAPE, gaussian_fill, halton_positions, scan_schedule
from aip_vtok.c1.tokens import grouped, receive, send, wire_cost
from aip_vtok.c1.training import Run, budgets_for, make_plans
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.diagnostics.diag0.io import PROJECTOR, Catalog, sealed_projector
from aip_vtok.runtime import file_hash
from aip_vtok.tokenizer.fsq import FSQ


@pytest.mark.parametrize("name", ["S1", "S2a", "S2b", "S3"])
def test_sparse_slots_nest_and_are_unique(name):
    full = scan_schedule(name, weights=(40, 3, 2, 1))
    assert len(full) == 8192 and len(set(full)) == len(full)
    assert all(0 <= slot < 4 * math.prod(SHAPE) for slot in full)
    for budget in BUDGETS:
        assert scan_schedule(name, weights=(40, 3, 2, 1), budget=budget) == full[:budget]
    area = math.prod(SHAPE)
    if name in ("S1", "S2a"):
        assert all(slot < area for slot in full[:510])
    if name == "S1":
        assert sum(slot < area for slot in full) == area
        assert all(area <= slot < 2 * area for slot in full[area:])
    if name == "S2b":
        assert all(p + area == q for p, q in zip(full[:510:2], full[1:510:2]))
    if name.startswith("S2"):
        assert [sum(slot // area == g for slot in full) for g in range(4)] == [4096, 2048, 1229, 819]


def test_halton_keeps_first_hit_and_completes_in_raster_order():
    # Base-2/base-3 Halton i=1..4 lands at 1,2,1,0 on a 2×2 grid.
    assert halton_positions(2, 2, attempts=4) == (1, 2, 0, 3)
    assert set(halton_positions(*SHAPE)) == set(range(math.prod(SHAPE)))


def test_receiver_rejects_an_altered_preshared_table():
    lock = {"winner": {"schedule": "S1", "p_coordinate": "aligned"}, "shape": list(SHAPE),
            "s3_static_weights": {"aligned": [1, 1, 1, 1]}, "slots": list(scan_schedule("S1"))}
    validate_scan_lock(lock)
    lock["slots"][1] = lock["slots"][0]
    with pytest.raises(ValueError, match="deterministic slot"):
        validate_scan_lock(lock)


def test_group_gaussian_preserves_observations_and_empty_groups():
    values = torch.zeros((1, 4, 6, 7, 8))
    mask = torch.zeros((1, 4, 7, 8), dtype=torch.bool)
    mask[:, 0, 2, 3] = True
    mask[:, 1, 4, 4] = True
    mask[:, 2] = True
    values[:, 0, :, 2, 3], values[:, 1, :, 4, 4] = 2, -3
    values[:, 2] = torch.randn_like(values[:, 2])
    expanded = mask[:, :, None].expand_as(values)
    out = gaussian_fill(values, mask, 1.5)
    assert torch.equal(out[expanded], values[expanded])
    assert torch.allclose(out[:, 0, :, 3, 3], torch.full((1, 6), 2.0), atol=1e-5)
    assert torch.allclose(out[:, 1, :, 3, 3], torch.full((1, 6), -3.0), atol=1e-5)
    assert out[:, 3].count_nonzero() == 0 and out.isfinite().all()
    assert torch.equal(gaussian_fill(values, torch.ones_like(mask), 1.5), values)


def test_physical_p_rank_fill_and_chain_rule():
    alignment = ChannelAlignment()
    alignment.p_mean.copy_(torch.linspace(-0.5, 0.5, 128).view_as(alignment.p_mean))
    alignment.p_std.copy_(torch.linspace(0.2, 3, 128).view_as(alignment.p_std))
    native = torch.randn(1, 128, 2, 3, requires_grad=True)
    aligned = alignment.align(native, False)
    objective = aligned.sin().square().mean()
    grad_aligned, = torch.autograd.grad(objective, aligned, retain_graph=True)
    grad_native, = torch.autograd.grad(objective, native)
    assert torch.allclose(native_gradient(grad_aligned, alignment), grad_native)
    mean = torch.linspace(1, 2, 128)
    fit = ChannelFit(mean, torch.ones(128), mean.square(), torch.ones(22, 6))
    ordered = fit.order_values(native.detach())
    top = fit.restore_rank(ordered[:, :24], "zero_fill")
    averaged = fit.restore_rank(ordered[:, :24], "train_mean_fill")
    assert torch.equal(top[:, fit.indices[:24]], averaged[:, fit.indices[:24]])
    assert top[:, fit.indices[24:]].count_nonzero() == 0
    assert torch.equal(averaged[:, fit.indices[24:], 0, 0], mean[fit.indices[24:]][None])
    assert not torch.equal(alignment.align(top, False), alignment.align(averaged, False))
    assert (alignment.align(fit.restore_rank(ordered, "zero_fill"), False) - aligned).abs().max() < 1e-5
    assert (alignment.restore(aligned, False) - native).abs().max() < 1e-5


def test_ids_are_budget_independent_and_only_receipt_is_observed():
    fsq = FSQ()
    values = grouped(torch.randn(1, 24, 3, 4))
    scale = torch.rand(4, 6) + 0.5
    slots = torch.tensor([0, 12, 47, 1, 15, 46])
    packet = send(fsq, values, scale, slots)
    received = receive(fsq, packet.payload[:, :3], scale, slots[:3], (3, 4), packet.version)
    assert received.mask.sum() == 3
    assert received.values[~received.mask[:, :, None].expand_as(received.values)].count_nonzero() == 0
    assert torch.equal(packet.payload[:, :3], send(fsq, values, scale, slots[:3]).payload)
    assert torch.equal(fsq.codes_to_ids(fsq.ids_to_codes(packet.payload)), packet.payload)
    ledger = wire_cost(510)
    assert ledger["scan_table_bytes"] == 0
    assert ledger["packet_count"] == 4 and ledger["header_bytes"] == 48
    assert ledger["raw_total_bytes"] == 1068
    assert wire_cost(1024)["packet_count"] == 9  # 510 is a packet boundary.


@pytest.mark.parametrize("intra", [True, False])
def test_recovery_identity_and_observed_values_are_frozen(intra):
    model = RecoveryNet(width=8)
    shape = (3, 4)
    base = torch.randn(1, 128, *shape)
    mask = torch.ones(1, 4, *shape, dtype=torch.bool)
    mask[:, 3, 2, 2] = False
    context = FrameContext(0 if intra else 1, intra, None if intra else torch.randn(1, 256, 6, 8), None)
    indices = torch.arange(24)
    assert torch.equal(model(base, mask, indices, context), base)
    nn.init.ones_(model.outputs["I" if intra else "P"].bias)
    result = model(base, mask, indices, context)
    expected = base.clone()
    expected[:, 24:] += 1
    expected[:, 18:24, 2, 2] += 1
    assert torch.equal(result, expected)


def test_same_context_native_teacher_and_detached_reference():
    class ReceiverContext(FrameContext):
        def commit_reference_once(self, feature, latent=None):
            assert not torch.is_grad_enabled()
            return super().commit_reference_once(feature, latent)

    class Decoder:
        def decode_native(self, y, context):
            assert context is current and y is native
            assert not torch.is_grad_enabled()
            return None, (y[:, :1] + context.context).expand(-1, 256, -1, -1)

    native = torch.full((1, 128, 2, 2), 3.0, requires_grad=True)
    latent = torch.ones(1, 128, 2, 2, requires_grad=True)
    memory = torch.randn(1, 4, 2, 2, requires_grad=True)
    current = ReceiverContext(1, False, torch.ones(1, 1, 2, 2), memory)
    projector = nn.Conv2d(128, 256, 1, bias=False).requires_grad_(False)
    nn.init.constant_(projector.weight, 1 / 128)
    loss, feature = reference_supervision(Decoder(), lambda y, intra: projector(y), latent, native, current)
    assert loss.item() == pytest.approx(3)
    loss.backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert native.grad is None and projector.weight.grad is None
    assert not current.committed
    state = commit_reference(current, latent, feature)
    for stored, source in ((state.latent, latent), (state.feature, feature), (state.memory, memory)):
        assert torch.equal(stored, source) and stored.data_ptr() != source.data_ptr()
        assert not stored.requires_grad and stored.grad_fn is None
    with pytest.raises(RuntimeError, match="already committed"):
        commit_reference(current, latent, feature)


def test_sealed_projector_digest():
    if not PROJECTOR.is_file():
        pytest.skip("requires the sealed local W128 projector")
    assert file_hash(PROJECTOR) == sealed_projector()["sha256"]


def test_calibration_videos_are_disjoint_from_fitting_and_gate0():
    rows = {str(i): {"original_frames": 64} for i in range(20)}
    train, cal = split_train(Catalog("train", rows, {}), ("0", "1"))
    assert len(cal.rows) == 5
    assert not (cal.rows.keys() & train.rows.keys())
    assert set(train.rows) | set(cal.rows) == set(rows)
    assert not {"0", "1"} & cal.rows.keys()
    with pytest.raises(ValueError, match="only on Train"):
        split_train(Catalog("validation", rows, {}), ())


def test_sparse_training_plan_uses_fixed_prefix_and_multiple_gop_phases():
    catalog = Catalog("train", {id_: {"original_frames": 72} for id_ in ("a", "b", "c")}, {})
    run = Run("c1b", 1, 120, 8, late_gop=True)
    plans = make_plans(catalog, run)
    assert plans == make_plans(catalog, run)
    assert {p["source"] for p in plans} == {0, 1, 2}
    assert {p["phase"] for p in plans} == {0, 4, 12, 20, 26}
    assert all(p["length"] == p["phase"] + 8 and p["start"] + p["length"] <= 72 for p in plans)
    rng = np.random.default_rng(1)
    assert budgets_for(SimpleNamespace(intra=True), True, rng) == (2048, 4096, 8192)
    p = budgets_for(SimpleNamespace(intra=False), True, rng)
    assert p[0] == 510 and p[1] in BUDGETS[1:-1] and p[2] == 8192
    assert budgets_for(SimpleNamespace(intra=False), False, rng) == (DENSE,)


def test_sparse_gate_rejects_incomplete_or_one_seed_dense_improvement():
    rows = [
        {"history": mode, "method": method, "seed": seed, "frames": 174,
         "psnr_global_mse": psnr, "lpips": lpips, "dists": 0.3}
        for mode in (INDEPENDENT, PAIRED)
        for method, seed, psnr, lpips in ((BASELINE, 0, 25.0, 0.5), (TRAINED, 1, 26.0, 0.4), (TRAINED, 2, 25.2, 0.4))
    ]
    _, passed = dense_decision(rows, 174)
    assert not passed
    manifest = {"completed": True, "steps_completed": {"1": 1000, "2": 1000}, "locked_sha256": "fixed"}
    phase1 = {"passes_for_c1b": True}
    with pytest.raises(RuntimeError, match="dual-seed"):
        require_sparse_release(phase1, {**manifest, "passes_for_c1b": passed}, "fixed")
    for row in rows:
        if row["seed"] == 2:
            row["psnr_global_mse"] += 1.0
    gains, passed = dense_decision(rows, 174)
    assert passed and len(gains) == 4
    require_sparse_release(phase1, {**manifest, "passes_for_c1b": passed}, "fixed")
    with pytest.raises(RuntimeError, match="Phase 1"):
        require_sparse_release({"passes_for_c1b": False}, {**manifest, "passes_for_c1b": passed}, "fixed")
    with pytest.raises(RuntimeError, match="same lock"):
        require_sparse_release(phase1, {**manifest, "passes_for_c1b": passed}, "different")


@pytest.mark.gpu
@pytest.mark.integration
def test_real_dmc_ld_rgb_gradient_and_projector_digest():
    if not torch.cuda.is_available() or not PROJECTOR.is_file():
        pytest.skip("requires CUDA and the sealed frozen assets")
    config, _ = frozen_inputs()
    digest = file_hash(PROJECTOR)
    codec = load_frozen(config)
    backbone = codec.backbone
    backbone.geometry = Geometry(64, 128)
    rgb = torch.rand(1, 3, 64, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(17))
    with torch.no_grad():
        ic = backbone.begin_frame(ReferenceState(), 0)
        y = backbone.aligned_latent(rgb, ic)
        pc = backbone.begin_frame(commit_reference(ic, y, codec.projector(y, True)), 1)
        y_native = backbone.native_latent(rgb, pc)
    y_native.requires_grad_()
    y_aligned = backbone.alignment.align(y_native.detach(), False).requires_grad_()
    assert (backbone.alignment.restore(y_aligned, False) - y_native).abs().max() < 1e-5
    restored_rgb, _ = backbone.decode_aligned(y_aligned, pc, checkpointed=True)
    native_rgb, _ = backbone.decode_native(y_native, pc, checkpointed=True)
    ga, = torch.autograd.grad((restored_rgb - rgb).square().mean(), y_aligned)
    gn, = torch.autograd.grad((native_rgb - rgb).square().mean(), y_native)
    torch.testing.assert_close(native_gradient(ga, backbone.alignment), gn, atol=2e-7, rtol=1e-3)
    loss, feature = reference_supervision(backbone, codec.projector, y_aligned, y_native, pc)
    loss.backward()
    assert y_aligned.grad.abs().sum() > 0 and y_native.grad is None
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    state = commit_reference(pc, y_aligned, feature)
    assert state.feature.data_ptr() != feature.data_ptr() and not state.feature.requires_grad
    assert file_hash(PROJECTOR) == digest
