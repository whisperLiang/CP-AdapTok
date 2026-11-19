from types import SimpleNamespace

import pytest
import torch

from aip_vtok.c1.recovery_net import reference_supervision
from aip_vtok.c1.scan import BUDGETS, SHAPE, halton_positions
from aip_vtok.c1_followup.quantization import ORIGINAL, Quantizer, Stage
from aip_vtok.decoder.causal_state import FrameContext
from aip_vtok.runtime import file_hash
from aip_vtok.sparse_rfsq.receiver import RecoveryNet, interpolate, sparse_values
from aip_vtok.sparse_rfsq.scan import Plan, scan_table


@pytest.mark.parametrize(("name", "ratio", "at_max"), [
    ("SP-1", 0, 0), ("SP-2", 0, 4096), ("SP-3", 20, 1536),
    ("SP-3", 30, 2304), ("SP-3", 50, 3841),
])
def test_nested_static_scan(name, ratio, at_max):
    weights = (21, 11, 7, 3)
    keys = scan_table(name, weights, ratio)
    assert keys == scan_table(name, weights, ratio)
    assert len(keys) == len(set(keys)) == 8192
    coarse = set()
    for level, group, y, x in keys:
        assert group in range(4) and y in range(68) and x in range(120)
        if level == 1:
            coarse.add((group, y, x))
        else:
            assert level == 2 and (group, y, x) in coarse
    assert len(coarse) == 8192 - at_max
    assert all(level == 1 for level, *_ in keys[:510])
    assert max(sum(g == group for _, g, *_ in keys[:510]) for group in range(4)) == 128
    if name == "SP-2":
        assert all(level == 1 for level, *_ in keys[:4096])
        assert keys[4096][1] == 0
    if name == "SP-3":
        for count in BUDGETS:
            assert sum(level == 2 for level, *_ in keys[:count]) == (count - 510) * ratio // 100
    plan = Plan.create(keys, SHAPE, 4, "cpu")
    ids = tuple(torch.arange(4 * 68 * 120).view(1, 4, 68, 120) + offset for offset in (0, 32640))
    payload = plan.encode(ids)
    for k in BUDGETS:
        torch.testing.assert_close(payload[:, :k], plan.encode_prefix(ids, k), rtol=0, atol=0)


def test_halton_deduplicates_then_completes_in_row_major_order():
    assert halton_positions(2, 3, attempts=2) == (1, 3, 0, 2, 4, 5)
    assert halton_positions(2, 3, attempts=6) == (1, 3, 2, 0, 4, 5)


@pytest.mark.parametrize("bad", [((2, 0, 1, 0),), ((1, 0, 1, 0), (1, 0, 1, 0)),
                                 ((2, 0, 1, 0), (1, 0, 1, 0))])
def test_plan_rejects_illegal_slots(bad):
    with pytest.raises(ValueError):
        Plan.create(bad, (2, 3), 4, "cpu")


def quantizer():
    return Quantizer(Stage([ORIGINAL] * 4, torch.ones(4, 6)),
                     Stage([ORIGINAL] * 4, torch.full((4, 6), 0.25), mu=2))


def test_decode_only_arrived_ids_and_compose_levels_exactly():
    q = quantizer()
    keys = [(1, 0, 1, 0), (1, 1, 0, 1), (2, 1, 0, 1)]
    plan = Plan.create(keys, (2, 3), 4, "cpu")
    sent = torch.tensor([[100, 199, 731], [0, 63999, 801]])
    got = plan.receive(q, sent)
    first = plan.receive(q, sent[:, :2])
    assert got.mask_l1.shape == got.mask_l2.shape == (2, 4, 2, 3)
    assert torch.equal(got.mask_l1, first.mask_l1) and not first.mask_l2.any()
    assert torch.equal(got.mask_l2.nonzero(), torch.tensor([[0, 1, 0, 1], [1, 1, 0, 1]]))
    assert (got.mask_l1 | ~got.mask_l2).all()
    ids1 = sent[:, 0, None, None, None].expand(-1, 4, 2, 3).clone()
    ids1[:, 1, 0, 1] = sent[:, 1]
    ids2 = sent[:, 2, None, None, None].expand_as(ids1)
    coarse, delta = q.primary.decode(ids1), q.residual.decode(ids2)
    assert torch.equal(got.values[~got.mask_l1], torch.zeros_like(got.values[~got.mask_l1]))
    assert torch.equal(got.values[0:2, 0, 1, 0], coarse[:, 0, 1, 0])
    assert torch.equal(got.values[got.mask_l2], (coarse + delta)[got.mask_l2])
    with pytest.raises(ValueError, match="Level 2 requires"):
        sparse_values(q, torch.stack((ids1, ids2), 1), torch.stack((~got.mask_l1, got.mask_l2), 1))
    with pytest.raises(ValueError):
        plan.receive(Quantizer(q.primary), sent)
    with pytest.raises(ValueError):
        plan.receive(q, sent + 64000)


def test_mask_normalized_gaussian_is_identity_on_observations():
    q = quantizer()
    values = torch.arange(4 * 5 * 6 * 6, dtype=torch.float32).reshape(1, 4, 5, 6, 6)
    ids, _ = q.encode(values / values.numel())
    masks = torch.ones_like(torch.stack(ids, dim=1), dtype=torch.bool)
    full = sparse_values(q, torch.stack(ids, 1), masks)
    assert torch.equal(interpolate(full, 1.5), full.values)
    masks[:, :, :, 1:] = False
    partial = sparse_values(q, torch.stack(ids, 1), masks)
    assert torch.equal(interpolate(partial, 1.5)[partial.mask_l1], partial.values[partial.mask_l1])
    empty = sparse_values(q, torch.stack(ids, 1), torch.zeros_like(masks))
    assert torch.equal(interpolate(empty, 1.5), empty.values)


@pytest.mark.parametrize("intra", [True, False])
def test_recoverynet_sees_both_masks_and_preserves_received_values(intra):
    model = RecoveryNet(width=4, context_channels=2)
    canvas = torch.zeros(1, 128, 4, 4)
    canvas[:, :6, 0, 0] = 0.125
    base = canvas.clone()
    base[:, :6, 1, 1] = 0.5
    primary = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    primary[:, 0, 0, 0] = True
    residual = torch.zeros_like(primary)
    ctx = FrameContext(0, intra, None if intra else torch.zeros(1, 2, 8, 8), None)
    top = torch.arange(24)
    assert torch.equal(model(canvas, primary, residual, top, ctx, base=base), base)
    with torch.no_grad():
        model.outputs["I" if intra else "P"].bias.fill_(1)
    seen = []
    handle = model.inputs["I" if intra else "P"].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    got = model(canvas, primary, residual, top, ctx, base=base)
    handle.remove()
    assert torch.equal(seen[0][:, :128], canvas)
    assert torch.equal(seen[0][:, 128:132].bool(), primary)
    assert torch.equal(seen[0][:, 132:136].bool(), residual)
    assert torch.equal(got[:, :6, 0, 0], canvas[:, :6, 0, 0])
    assert torch.equal(got[:, :6, 1, 1], base[:, :6, 1, 1] + 1)


def test_validation_and_test_pixel_access_is_fail_closed():
    from aip_vtok.c1_followup.io import ReadAudit

    audit = ReadAudit({})
    touched = []
    for split in ("validation", "test"):
        catalog = SimpleNamespace(split=split, read=lambda *args: touched.append(args))
        with pytest.raises(ValueError):
            audit.open(split)
        with pytest.raises(ValueError):
            audit.read(catalog, "sequence", 0, 4)
    audit.validation_locked = True
    with pytest.raises(ValueError):
        audit.open("test")
    assert not touched
    assert audit.counts == {"train": 0, "validation": 0, "test": 0}


class TinyProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, latent, intra):
        return (latent * self.scale).expand(-1, 192 if intra else 256, -1, -1)


@pytest.fixture
def toy_link(monkeypatch):
    from aip_vtok.sparse_rfsq import rollout

    def begin(state, index):
        intra = state.is_intra(index, 32)
        return FrameContext(index, intra, state.feature, state.feature)

    def capture(backbone, state, index, image):
        ctx = backbone.begin_frame(state, index)
        code = 1 if ctx.intra else int(ctx.context[0, 0, 0, 0])
        return SimpleNamespace(context=ctx, frame_type="I" if ctx.intra else "P", code=code, rgb=None)

    def gaussian(frame, backbone, context, payload):
        return payload.float().mean(dim=1, keepdim=True).reshape(1, 1, 1, 1)

    def b3(*args):
        return gaussian(*args) + 1

    def encode(sample):
        return torch.arange(8192, dtype=torch.long)[None] + sample.code

    monkeypatch.setattr(rollout, "capture", capture)
    case = SimpleNamespace(encode=encode, rates={k: {"payload_bits": 16 * k, "entropy_payload_bits": 8 * k}
                                                for k in BUDGETS})
    codec = SimpleNamespace(backbone=SimpleNamespace(begin_frame=begin), projector=TinyProjector())
    return rollout, codec, {"I": case, "P": case}, gaussian, b3


def test_independent_ref_comes_from_b3_and_display_budget_cannot_change_it(toy_link):
    rollout, codec, cases, gaussian, b3 = toy_link
    base, short, long = (rollout.CausalPair() for _ in range(3))
    for index in range(2):
        shared, a, b = (link.advance(codec, None, index, cases, receiver)
                        for link, receiver in ((base, gaussian), (short, b3), (long, b3)))
        case = cases[a.sample.frame_type]
        lo = b3(case, codec.backbone, a.context, a.payload[:, :2048])
        hi = b3(case, codec.backbone, b.context, b.payload[:, :8192])
        assert not torch.equal(lo, hi)
        assert short.rx.fingerprint() == long.rx.fingerprint() == long.tx.fingerprint()
        ref = 2048 if index == 0 else 510
        expected = b3(case, codec.backbone, a.context, a.payload[:, :ref])
        assert torch.equal(short.rx.latent, expected)
        assert torch.equal(short.rx.feature, codec.projector(expected, index == 0))
        assert not short.rx.latent.requires_grad and short.rx.latent.data_ptr() != expected.data_ptr()
        assert not torch.equal(base.rx.feature[:1], short.rx.feature[:1])
    assert a.sample.code != shared.sample.code
    assert not torch.equal(a.context.context, shared.context.context)


def test_evaluation_scores_b3_against_both_actual_histories(toy_link, monkeypatch):
    from aip_vtok.sparse_rfsq import evaluation as ev
    from aip_vtok.sparse_rfsq.scan import LEGACY

    _, codec, cases, gaussian, b3 = toy_link
    anchor = ev.Trial(LEGACY, "B2", cases, gaussian)
    learned = ev.Trial("SP-1", "B3", cases, b3, "W1")

    def score(codec, frame, reference, scorer, video, trial, history):
        context = frame.context
        case = trial.cases[frame.sample.frame_type]
        for k in BUDGETS:
            rec = trial.receiver(case, codec.backbone, context, frame.payload[:, :k])
            yield {**trial.fields(history), **reference, "frame": context.frame_id, "video": video,
                   "budget": k, "decoded": rec.item()}

    monkeypatch.setattr(ev, "score_frame", score)
    monkeypatch.setattr(ev, "FrameMetrics", lambda *_: None)
    rows = list(ev.video_rows(codec, None, [None, None], "test-video", anchor, [anchor, learned]))
    ev.validate_inventory(rows, {"test-video": 2}, [anchor, learned])
    b3_rows = {(r["history"], r["budget"]): r for r in rows if r["frame"] == 1 and r["method"] == "B3"}
    for k in BUDGETS:
        fixed = b3_rows["fixed_reference_B3", k]
        independent = b3_rows["independent_closed_B3", k]
        assert fixed["reference_method"] == "B2" and independent["reference_method"] == "B3"
        assert fixed["decoded"] != independent["decoded"]
        assert independent["reference_latent_rms"] == b3_rows["independent_closed_B3", 2048]["reference_latent_rms"]


def test_gradient_diagnostics_match_known_norms_and_cosines():
    from aip_vtok.sparse_rfsq.training import gradient_vectors, summarize_gradients

    p = torch.nn.Parameter(torch.ones(2))
    terms = {"mse": p[0] * 2, "lpips": p[0] * -3, "anchor": p[1] * 5}
    result = summarize_gradients(gradient_vectors(terms, (p,)))
    assert result["norms"] == {"mse": 2.0, "lpips": 3.0, "anchor": 5.0}
    assert result["cosines"] == {"mse/lpips": -1.0, "mse/anchor": 0.0, "lpips/anchor": 0.0}
    assert p.grad is None


def test_reference_supervision_reaches_recovery_but_projector_stays_frozen(tmp_path):
    p = TinyProjector()
    path = tmp_path / "projector.pt"
    torch.save(p.state_dict(), path)
    before = file_hash(path)
    context = FrameContext(0, True, None, None)
    native = torch.zeros(1, 192, 1, 1)
    backbone = SimpleNamespace(decode_native=lambda value, ctx: (None, value))
    latent = torch.ones(1, 1, 1, 1, requires_grad=True)
    loss, _ = reference_supervision(backbone, p, latent, native, context)
    loss.backward()
    assert latent.grad.item() == 2.0
    assert p.scale.grad is None and p.scale.item() == 2.0 and file_hash(path) == before


def test_calibration_deterministically_balances_families_and_freezes_one_ratio():
    from itertools import product

    from aip_vtok.sparse_rfsq.calibration import KINDS, SIGMAS, TRIALS, select

    rows = []
    for (strategy, pct), sigma, (index, k), kind in product(TRIALS, SIGMAS, enumerate(BUDGETS), KINDS):
        quality = 20 + index + (sigma == 2.5) + (pct in (20, 30))
        rows.append({"strategy": strategy, "refinement_pct": pct, "sigma": sigma, "budget": k,
                     "frame_type": kind, "mse": 10 ** (-quality / 10)})
    selected = select(rows)
    assert selected["sigma"] == 2.5 and selected["sp3_refinement_pct"] == 20 and selected["winner"] == "SP-3"
    assert select([row | {"mse": 0.01} for row in rows])["winner"] is None


def test_training_catalog_excludes_holdout_videos_and_duplicate_pixels():
    from aip_vtok.diagnostics.diag0.io import Catalog
    from aip_vtok.sparse_rfsq.calibration import training_catalog

    def row(*hashes):
        return {"original_frames": len(hashes), "frame_hashes": [{"sha256": h} for h in hashes]}

    train = Catalog("train", {"a": row("p", "q"), "b": row("r", "q"), "c": row("s")}, {})
    assert set(training_catalog(train, [{"video": "a"}]).rows) == {"c"}


@pytest.mark.parametrize(("gain", "lpips", "dists", "psnr_ok", "perceptual_ok"), [
    (0.5, -0.01, 0.005, True, True),
    (0.49, -0.01, 0.005, False, True),
    (0.5, 0.00, 0.005, True, False),
    (0.5, -0.01, 0.006, True, False),
])
def test_s2_focus_gate_checks_all_three_locked_budgets(gain, lpips, dists, psnr_ok, perceptual_ok):
    from aip_vtok.sparse_rfsq.gates import focus_checks

    rows = [{"history": "independent_closed", "budget": k, "psnr_gain_db": g, "lpips_change": lpips,
             "dists_change": dists} for k, g in zip((510, 2048, 8192), (gain, gain, 0.1))]
    assert focus_checks(rows) == {"focus_psnr": psnr_ok, "focus_lpips_dists": perceptual_ok}


def test_s2_withheld_without_opening_train(tmp_path, monkeypatch):
    from aip_vtok.sparse_rfsq import training

    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s1.write_text('{"released": false}')
    monkeypatch.setattr(training, "S1_GATE", s1)
    monkeypatch.setattr(training, "S2_GATE", s2)
    monkeypatch.setattr(training, "load_s1", lambda _: {"released": False, "reason": "veto"})
    training.run_pilot(SimpleNamespace(), {}, {})
    assert '"started": false' in s2.read_text()


def test_metric_reuse_requires_equal_decoder_input_and_target(monkeypatch):
    from aip_vtok.sparse_rfsq import evaluation as ev

    calls = []

    def decode(x, context):
        calls.append(context.context.item())
        return x + context.context, None

    def frame(context, target):
        return SimpleNamespace(context=SimpleNamespace(context=torch.tensor(context)),
                               sample=SimpleNamespace(aligned=torch.tensor(target), rgb=None))

    codec = SimpleNamespace(backbone=SimpleNamespace(decode_aligned=decode))
    monkeypatch.setattr(ev, "FrameMetrics", lambda *_: lambda rec, _, target: {"mse": (rec - target).square().item()})
    monkeypatch.setattr(ev, "chroma_planes", lambda *_: {})
    scorer = ev.FrameScorer(codec, None, None)
    x = torch.tensor(0.0)
    for ctx, target, latent, expected in [(1, 0, x, 1), (1, 0, x.clone(), 1), (2, 0, x, 4), (1, 1, x, 0),
                                          (1, 0, x + 2, 9)]:
        sample = frame(ctx, target)
        assert scorer.measure(sample, latent, scorer.context_cache(sample)[510]) == {"mse": expected}
    assert calls == [1, 2, 1, 1]


def test_payload_ledger_separates_fixed_bits_and_empirical_entropy():
    from aip_vtok.sparse_rfsq.scan import rates

    keys = ((1, 0, 0, 0), (1, 1, 0, 1), (2, 1, 0, 1))
    result = rates(keys, ((5.0, 4.0), (2.0, 3.0)), (1, 2, 3))
    assert [row["payload_bits"] for row in result.values()] == [16, 32, 48]
    assert [row["entropy_payload_bits"] for row in result.values()] == [5.0, 9.0, 12.0]
    assert [row["level2_symbols"] for row in result.values()] == [0, 0, 1]
    assert all(row["header_bytes"] > 0 and row["scan_table_bytes"] == 0 and not row["real_bitstream"]
               for row in result.values())
