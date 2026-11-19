"""Contracts for the independent C1 follow-up and its gated entry points."""

import json
from types import SimpleNamespace

import pytest
import torch

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.c1.channels import ALIGNED, ChannelFit
from aip_vtok.c1.tokens import dense_roundtrip
from aip_vtok.c1_followup import experiment, sparse, training
from aip_vtok.c1_followup.calibration import calibration_plan, sealed_training_plans, used_frame_hashes
from aip_vtok.c1_followup.dense import Layout, check_replay, quality_summary
from aip_vtok.c1_followup.gates import (
    HISTORIES,
    LIMITS,
    METHODS,
    admit_on_train,
    release_dense,
    sparse_release,
    training_gate,
)
from aip_vtok.c1_followup.io import LEGACY, ReadAudit, sealed_controls, tensor_digest
from aip_vtok.c1_followup.quantization import ORIGINAL, Quantizer, Stage, dense_ledger
from aip_vtok.c1_followup.rollout import recover, summarize_rollout
from aip_vtok.diagnostics.diag0.io import PROJECTOR, Catalog, json_at
from aip_vtok.runtime import file_hash
from aip_vtok.tokenizer.fsq import FSQ
from aip_vtok.training.fix import fix_settings


@pytest.fixture
def values():
    return torch.randn(1, 4, 5, 7, 6, generator=torch.Generator().manual_seed(31))


def stage(values, levels=ORIGINAL, mu=0, scale=0.75):
    return Stage([levels] * values.shape[1], torch.full((values.shape[1], 6), scale), mu)


def test_a1_quantizer_matches_sealed_c1_bits_and_decode(values):
    quant = Quantizer(stage(values))
    old, packet = dense_roundtrip(FSQ(), values.movedim(-1, 2), quant.primary.scales)
    ids, saturation = quant.encode(values)
    assert torch.equal(ids[0], packet.payload)
    assert torch.equal(saturation[0], packet.saturated)
    assert torch.equal(quant.decode(*ids), old.movedim(2, -1))


@pytest.mark.parametrize("levels", [ORIGINAL, (10, 8, 8, 5, 5, 4), (10, 10, 8, 5, 4, 4)])
def test_every_codeword_roundtrips(levels):
    ids = torch.arange(64_000).reshape(1, 1, 1, -1)
    fsq = Stage([levels], torch.ones(1, 6), 0).fsq
    assert torch.equal(fsq.codes_to_ids(fsq.ids_to_codes(ids)), ids)
    with pytest.raises(ValueError, match="ID outside"):
        fsq.ids_to_codes(torch.full_like(ids[:, :, :, :1], 64_000))


@pytest.mark.parametrize("mu", [0, 2, 8])
def test_residual_is_strictly_decoded_from_both_integer_ids(values, mu):
    first = stage(values, mu=mu)
    second = stage(values, levels=(10, 8, 8, 5, 5, 4), mu=mu, scale=0.45)
    quant = Quantizer(first, second)
    ids, saturation = quant.encode(values)
    reconstructed = quant.decode(*ids)
    first_value = first.decode(ids[0])
    expected_id, expected_sat = second.encode(values - first_value)
    assert torch.equal(ids[1], expected_id)
    assert torch.equal(saturation[1], expected_sat)
    assert torch.equal(reconstructed, first_value + second.decode(ids[1]))
    assert torch.equal(first_value, quant.decode(ids[0]))
    with pytest.raises(ValueError, match="Level 1"):
        quant.decode(None, ids[1])
    with pytest.raises(ValueError, match="same slots"):
        quant.decode(ids[0], ids[1][..., :1])
    with pytest.raises(ValueError, match="single-stage"):
        Quantizer(first).decode(*ids)
    assert not reconstructed.requires_grad


def test_invalid_alphabet_or_scale_is_rejected():
    with pytest.raises(ValueError, match="64,000"):
        Stage([(4,) * 6], torch.ones(1, 6), 0)
    with pytest.raises(ValueError, match="positive"):
        Stage([ORIGINAL], torch.zeros(1, 6), 0)


@pytest.mark.parametrize("groups,stages,bits", [(4, 1, 64), (4, 2, 128), (6, 1, 96), (8, 1, 128)])
def test_wire_ledger_counts_payload_and_each_level_header(groups, stages, bits):
    ledger = dense_ledger(groups, stages, (68, 120), 1.0)
    slots = groups * 68 * 120
    packets = ((slots + 127) // 128) * stages
    assert ledger["payload_bits_per_position"] == bits
    assert ledger["payload_bits"] == bits * 68 * 120
    assert ledger["header_bytes"] == 16 + 9 * packets
    assert ledger["raw_total_bytes"] == ledger["payload_bits"] // 8 + ledger["header_bytes"]
    assert ledger["entropy_est_bits"] == ledger["header_bytes"] * 8 + slots * stages
    assert ledger["real_bitstream"] is False


def test_replay_matches_both_sealed_cohorts_and_rejects_mixing(tmp_path):
    rows = [
        {"arm": arm, "video": "v", "frame": frame, "frame_type": kind, "mse": error}
        for arm, error in [("A0", 0.001), ("A1", 0.002)]
        for frame, kind in enumerate(("I", "P"))
    ]
    old = tmp_path / "history.json"
    old.write_text(json.dumps({"rows": [
        {**row, "history": "same_context_b0_fsq_history",
         "method": {"A0": "B0-cont", "A1": "B0-fsq"}[row["arm"]], "seed": 0}
        for row in rows
    ]}))
    assert check_replay(rows, old)["passed"]
    assert quality_summary(rows[:2])["I"]["frames"] == 1
    assert not check_replay(rows[:-1], old)["passed"]
    assert not check_replay([row | {"mse": 0.1} for row in rows], old)["passed"]


def quality(db=27.6, lpips=0.32, dists=0.17, ms=1):
    return {"psnr_global_mse": db, "lpips": lpips, "dists": dists, "encode_ms": ms, "decode_ms": ms}


def test_validation_can_veto_but_cannot_reselect_train_shortlist():
    train = {"A1": quality(), "A2": quality(28), "A3": quality(30, ms=2), "A5-fsq": quality(28.3)}
    shortlist = admit_on_train(train, LIMITS)
    assert set(shortlist) == {"A2", "A3"}
    val = train | {"A2": quality(29, dists=.18), "A3": quality(28, ms=2), "A5-fsq": quality(31)}
    admission = release_dense(shortlist, val, LIMITS)
    assert admission["released"] == []
    assert set(admission["candidates"]) == {"A2", "A3"}


def gate_table(psnr_gain=0.6, lpips=0.31, dists=0.175, sync=True):
    return {
        name: {"B0-quantized": quality(), "B1-quantized": quality(27.6 + gain, lpips, dists),
               "state_sync": sync}
        for name, gain in (("same_context", psnr_gain), ("independent_closed", psnr_gain))
    }


@pytest.mark.parametrize("milestone", [500, 1000])
def test_training_gate_uses_both_histories_and_finite_synchronized_values(milestone):
    assert training_gate(gate_table(), milestone, LIMITS)["passed"]
    assert not training_gate(gate_table(sync=False), milestone, LIMITS)["passed"]
    assert not training_gate(gate_table(lpips=0.32), milestone, LIMITS)["passed"]
    assert not training_gate(gate_table(dists=0.176), milestone, LIMITS)["passed"]
    assert not training_gate(gate_table(psnr_gain=float("nan")), milestone, LIMITS)["passed"]
    weak = gate_table()
    weak["independent_closed"] = gate_table(psnr_gain=-0.01)["independent_closed"]
    assert not training_gate(weak, milestone, LIMITS)["passed"]


def test_sealed_test_boundary_and_projector_control():
    audit = ReadAudit({})
    with pytest.raises(ValueError, match="pixel"):
        audit.open("test")
    assert audit.counts["test"] == 0
    assert audit.denied == 1
    controls = sealed_controls()
    assert controls["legacy_c1b_steps"] == {"seed1": 0, "seed2": 0}
    assert controls["projector"]["sha256"] == "3b1bd715427bd177393e9552a99576e614ab7dd1ed307155129cf62896d5eb94"


def test_a0_continuous_restores_the_same_zero_filled_rank():
    torch.manual_seed(44)
    latent = torch.randn(1, 128, 4, 5)
    fit = ChannelFit(torch.zeros(128), torch.ones(128), torch.rand(128), torch.ones(22, 6))
    ordered = fit.order_values(latent)[:, :24]
    layout = Layout(fit.indices[:24], 128, ALIGNED)
    sample = SimpleNamespace(aligned=latent, native=latent)
    payload, saturation = layout.encode(sample)
    assert not saturation
    assert torch.equal(layout.decode(payload, None), fit.restore_rank(ordered))


@pytest.mark.parametrize("groups", [4, 6, 8])
def test_frozen_projector_gradients_reach_recovery_weights_through_integer_decode(groups):
    torch.manual_seed(17)
    expected_sha = sealed_controls()["projector"]["sha256"]
    artifact = torch.load(PROJECTOR, map_location="cpu", weights_only=True)
    projector = LatentToRef(artifact["width"]).eval().requires_grad_(False)
    projector.load_state_dict(artifact["projector"])
    before = tensor_digest(projector)
    model = training.recovery_model(groups, "cpu")
    ctx = SimpleNamespace(intra=True)
    indices = torch.randperm(128)[:groups * 6]
    source = torch.randn(1, 128, 4, 5, requires_grad=True)
    quant = Quantizer(Stage([ORIGINAL] * groups, torch.ones(groups, 6)),
                      Stage([ORIGINAL] * groups, torch.full((groups, 6), 0.2), mu=2))
    layout = Layout(indices, 128, ALIGNED, quant)
    ids, _ = layout.encode(SimpleNamespace(native=source, aligned=source))
    aligned = recover(layout, None, ids, model, ctx)
    assert torch.equal(aligned, layout.restore(quant.decode(*ids), None))
    projector(aligned, True).square().mean().backward()
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(torch.tensor(norms))) and max(norms) > 0
    assert source.grad is None and all(p.grad is None for p in projector.parameters())
    assert before == tensor_digest(projector)
    assert expected_sha == file_hash(PROJECTOR)


def test_train_holdout_has_no_frame_overlap_and_validation_requires_lock():
    config = fix_settings("fix_a", 1)
    reader = ReadAudit(config)
    with pytest.raises(ValueError, match="locked Validation"):
        reader.open("validation")
    catalog = reader.open("train")
    original = json_at(LEGACY / "scan_lock.json")["fit_plan"]
    plans = sealed_training_plans(catalog, LIMITS)
    holdout = calibration_plan(catalog, original, plans, config["calibration"]["videos"])
    videos = sorted(catalog.rows)
    fitting = [(row["video"], row["start"], row["frames"]) for row in original]
    optimization = [(videos[row["source"]], row["start"], row["length"]) for plan in plans.values() for row in plan]
    reserved = [(row["video"], row["start"], row["frames"]) for row in holdout]
    assert len(holdout) >= 5
    assert used_frame_hashes(catalog, reserved).isdisjoint(used_frame_hashes(catalog, fitting + optimization))
    assert set(v for v, _, _ in reserved).isdisjoint(config["calibration"]["videos"])
    with pytest.raises(ValueError, match="requires Train"):
        calibration_plan(Catalog("validation", {}, {}), original, plans, [])
    assert reader.counts == {"train": 0, "validation": 0, "test": 0}


@pytest.mark.parametrize("passes, expected", [
    ([False], [(1, 500)]),
    ([True, False], [(1, 500), (1, 1000)]),
    ([True, True, False], [(1, 500), (1, 1000), (2, 1000)]),
    ([True, True, True], [(1, 500), (1, 1000), (2, 1000)]),
])
def test_candidate_advances_only_on_the_prior_gate(monkeypatch, passes, expected):
    done = []
    integrity = {"passed": True, "test_images_read": 0}
    run = SimpleNamespace(verify_integrity=lambda: integrity)

    def milestone(run, train, anchor, layouts, seed, target):
        done.append((seed.seed, target))
        return {"step": target, "seed": seed.seed, "gate": {"passed": passes[len(done) - 1]}}

    monkeypatch.setattr(training, "run_milestone", milestone)
    result = training.run_candidate(run, None, None, None, [SimpleNamespace(seed=1), SimpleNamespace(seed=2)])
    assert done == expected
    assert result["sparse_released"] == (len(passes) == 3 and all(passes))
    assert result["steps"] == {
        f"seed{seed}": max((step for s, step in expected if s == seed), default=0) for seed in (1, 2)
    }


def test_sparse_release_fails_closed_on_wrong_seed_step_or_integrity():
    integrity = {"passed": True, "test_images_read": 0}
    first = {"step": 1000, "seed": 1, "gate": {"passed": True}}
    second = {**first, "seed": 2}
    assert sparse_release(True, first, second, integrity, LIMITS)
    assert not sparse_release(False, first, second, integrity, LIMITS)
    assert not sparse_release(True, first, second | {"step": 500}, integrity, LIMITS)
    assert not sparse_release(True, second, first, integrity, LIMITS)
    assert not sparse_release(True, first, second, integrity | {"test_images_read": 1}, LIMITS)
    assert not sparse_release(True, first, second, integrity | {"passed": False}, LIMITS)


def test_closed_sparse_gate_writes_only_negative_record(tmp_path, monkeypatch):
    phase1 = tmp_path / "phase1.json"
    phase1.write_text("{}")
    dense = tmp_path / "training.json"
    dense.write_text(json.dumps({"phase1_sha256": file_hash(phase1), "sparse_released": []}))
    result = tmp_path / "sparse_gate.json"
    scans = tmp_path / "c1b_followup"
    monkeypatch.setattr(sparse, "PHASE1", phase1)
    monkeypatch.setattr(sparse, "TRAINING_GATE", dense)
    monkeypatch.setattr(sparse, "SPARSE_GATE", result)
    monkeypatch.setattr(sparse, "SPARSE", scans)
    monkeypatch.setattr(sparse, "load_phase1", lambda: {"released": ["A3"]})
    run = SimpleNamespace(verify_integrity=lambda: {"passed": True, "test_images_read": 0})
    sparse.run_sparse(run)
    gate = json_at(result)
    assert gate["label"] == "c1b_followup" and gate["started_validation"] is False
    assert gate["training_steps"] == 0 and not scans.exists()


@pytest.mark.parametrize("groups", [6, 8])
def test_sparse_scan_unique_by_group_and_prefix_exact_including_halton_completion(groups):
    shape = (3, 7)
    weights = tuple(range(1, groups + 1))
    slots = tuple(sparse.scan_slots(shape, weights))
    assert slots == tuple(sparse.scan_slots(shape, weights))
    assert len(set(slots)) == groups * 3 * 7
    assert len({(y, x) for _, y, x in slots}) == 3 * 7
    assert tuple(sparse.scan_slots(shape, weights, 17)) == slots[:17]
    assert sparse.halton_positions(*shape, attempts=1)[1:] == tuple(
        i for i in range(21) if i != sparse.halton_positions(*shape, attempts=1)[0]
    )


def test_sparse_progressive_truncation_and_gaussian_preserve_observed_values(values):
    quantizer = Quantizer(stage(values), stage(values, scale=0.17))
    ids, _ = quantizer.encode(values)
    shape = values.shape[2:4]
    schedule = tuple(sparse.progressive_slots(sparse.scan_slots(shape, (1, 2, 3, 4)), 2))
    plan = sparse.SparsePlan.create(schedule, shape, 4, "cpu")
    full = plan.encode_prefix(ids, len(schedule))
    for n in (0, 1, 127, 128, 129, len(schedule)):
        partial = plan.encode_prefix(ids, n)
        assert torch.equal(partial, full[:, :n])
        received, mask = plan.values_and_mask(quantizer, partial)
        restored = sparse.interpolate(received, mask)
        assert torch.equal(restored[mask], received[mask])
        assert torch.isfinite(restored).all()
        if n == 0:
            assert not restored.any()
        if n == len(schedule):
            assert torch.equal(restored, quantizer.decode(*ids))
    single = torch.zeros_like(values[..., 0], dtype=torch.bool)
    single[:, 1, 2, 3] = True
    filled = sparse.interpolate(values, single)
    assert not filled[:, (0, 2, 3)].any()
    assert torch.equal(filled[single], values[single])


@pytest.mark.parametrize("schedule, reason", [
    ([(2, 0, 0, 1)], "earlier Level 1"),
    ([(1, 0, 0, 1), (1, 0, 0, 1)], "duplicate"),
    ([(1, 0, 0, 1), (2, 1, 0, 1)], "earlier Level 1"),
])
def test_sparse_rejects_missing_dependency_or_duplicate_slots(schedule, reason):
    with pytest.raises(ValueError, match=reason):
        sparse.SparsePlan.create(schedule, (2, 2), 2, "cpu")


def test_sparse_budget_caps_charge_all_transmitted_levels():
    table = sparse.scan_slots((34, 60), (1, 1, 1, 1))
    budgets = list(sparse.budget_prefixes(sparse.progressive_slots(table, 2), [[9] * 4, [8] * 4]))
    target = {x["budget_type"]: x for x in budgets if x["budget"] == 510}
    assert target["fixed_slots"]["unique_slots"] == 510
    assert target["fixed_slots"]["level2_symbols"] > 0
    assert target["fixed_bits"]["raw_bits"] <= target["fixed_bits"]["cap"]
    assert target["train_entropy_budget"]["train_entropy_est_bits"] <= target["train_entropy_budget"]["cap"]
    assert target["fixed_bits"]["prefix_symbols"] < target["train_entropy_budget"]["prefix_symbols"]


def test_milestone_requires_each_unique_frame_in_both_histories():
    rows = [
        {"video": "test_fixture", "frame": frame, "frame_type": ("I", "P")[frame],
         "mse": .002, "lpips": .3, "dists": .1, "history": history, "method": method, "state_sync": True}
        for history in HISTORIES for method in METHODS for frame in range(2)
    ]
    cohort = {"test_fixture": 2}
    assert summarize_rollout(rows, cohort)["same_context"]["B0-cont"]["frames"] == 2
    with pytest.raises(RuntimeError, match="every unique frame"):
        summarize_rollout(rows[:-1] + [rows[-2]], cohort)


def test_frozen_candidate_lock_checks_train_evidence_and_pixel_counts(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    calibration = tmp_path / "calibration.json"
    train = {"shortlist": ["A2"], "arms": {"A2": {}}}
    calibration.write_text(json.dumps(train))
    lock = tmp_path / "candidates.json"
    contract = {**train, "protocol_sha256": file_hash(protocol), "calibration_sha256": file_hash(calibration),
                "validation_pixels_read_before_lock": 0, "test_images_read": 0}
    lock.write_text(json.dumps(contract))
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment, "PROTOCOL", protocol)
    monkeypatch.setattr(experiment, "CANDIDATES", lock)
    assert experiment.frozen_candidates() == contract
    lock.write_text(json.dumps(contract | {"validation_pixels_read_before_lock": 1}))
    with pytest.raises(RuntimeError, match="Train-only"):
        experiment.frozen_candidates()
    lock.write_text(json.dumps(contract | {"shortlist": ["A3"]}))
    with pytest.raises(RuntimeError, match="changed"):
        experiment.frozen_candidates()


def test_report_renders_unstarted_seed_and_final_steps():
    from aip_vtok.c1_followup.report import render_actual_steps

    lines = render_actual_steps({"candidates": {"A2": {
        "steps": {"seed1": 500, "seed2": 0}, "pilot": {"gate": {"passed": False}},
        "seed1_final": None, "seed2_final": None,
    }}})
    assert "| A2 | 500 | 0 | 未通过 | 未启动 | 未启动 |" in lines
