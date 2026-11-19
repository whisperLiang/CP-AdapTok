import os
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from aip_vtok.c1.recovery_net import reference_supervision
from aip_vtok.c1.scan import BUDGETS, REF_BUDGET
from aip_vtok.c1_followup.io import tensor_digest
from aip_vtok.diagnostics.diag0.closed import same_state
from aip_vtok.runtime import write_json
from aip_vtok.sparse_decoder.evaluation import CLOSED, FIXED, validate_rows
from aip_vtok.sparse_decoder.models import DecoderReceiver, create_model
from aip_vtok.sparse_decoder.phase0 import contrasts
from aip_vtok.sparse_decoder.protocol import OUTPUT, SIGMA, open_run
from aip_vtok.sparse_rfsq.receiver import Receiver
from aip_vtok.sparse_rfsq.rollout import CausalPair


def test_factorial_effects_are_oriented_and_include_history():
    def point(value):
        return {"psnr_global_mse": value, "psnr_frame_mean": 2 * value,
                "lpips": -value, "dists": value / 10}

    fixed = dict(zip(("Q1", "Q2", "Q3", "Q4"), map(point, (10, 7, 11, 9))))
    closed = dict(zip(("Q1", "Q2", "Q3", "Q4"), map(point, (10, 6, 12, 8))))
    actual = contrasts(fixed, closed)
    assert actual[FIXED]["scan_a1"]["psnr_global_mse"] == -3
    assert actual[FIXED]["quant_old"]["lpips"] == -1
    assert actual[FIXED]["quant_sp1"]["dists"] == pytest.approx(0.2)
    assert actual[FIXED]["interaction"]["psnr_global_mse"] == 1
    assert actual[CLOSED]["interaction"]["psnr_frame_mean"] == 0
    assert actual["closed_minus_fixed"]["Q4_minus_Q1"]["psnr_global_mse"] == -1


def test_inventory_rejects_dropped_duplicate_and_unverified_rows():
    row = {"video": "clip", "frame": 0, "arm": "Q1", "history": FIXED, "budget": 510,
           "state_sync": True, "prefix_ids_match": True}
    expected = {("Q1", FIXED, 510)}
    validate_rows([row], {"clip": 1}, expected)
    for rows in ([], [row, row], [row | {"prefix_ids_match": False}], [row | {"frame": 1}]):
        with pytest.raises(RuntimeError):
            validate_rows(rows, {"clip": 1}, expected)


ARM_NAMES = ("E2", "E3", "E4")


@pytest.fixture(scope="module")
def native_run():
    run = open_run(metrics=False)
    try:
        yield run
    finally:
        evidence = run.verify_integrity()
        if os.environ.get("SPARSE_DECODER_TEST_AUDIT"):
            write_json(OUTPUT / "phase1_test_inputs.json", evidence)


@pytest.fixture(scope="module")
def native_frames(native_run):
    train = native_run.reader.open("train")
    holdout = next(row for row in native_run.lock["holdout"]
                   if train.rows[row["video"]]["original_frames"] >= 33)
    clip = native_run.reader.read(train, holdout["video"], 0, 33)
    cases = native_run.inputs.cases("SP-1", native_run.lock["keys"]["SP-1"])
    return clip, cases


@pytest.fixture(scope="module")
def two_frames(native_run, native_frames):
    clip, cases = native_frames
    pair = CausalPair()
    return [pair.advance(native_run.codec, clip[i], i, cases, Receiver(SIGMA)) for i in (0, 1)]


def receiver_for(arm):
    return DecoderReceiver(SIGMA, create_model(arm, "cuda"))


@pytest.mark.parametrize("arm", ARM_NAMES)
@torch.no_grad()
def test_t01_t03_t10_t14_step_zero_shapes_and_hard_observations(native_run, native_frames, two_frames, arm):
    backbone = native_run.codec.backbone
    cases = native_frames[1]
    receiver = receiver_for(arm)
    for frame in two_frames:
        case = cases[frame.sample.frame_type]
        for k in BUDGETS:
            prefix = frame.payload[:, :k]
            initial = receiver(case, backbone, frame.context, prefix)
            base = Receiver(SIGMA)(case, backbone, frame.context, prefix)
            assert torch.equal(initial, base)
            assert initial.shape == (1, 128, 68, 120) and initial.dtype == torch.float32
            assert torch.isfinite(initial).all() and prefix.dtype == torch.int64

    for head in receiver.model.outputs.values():
        head.bias.fill_(0.125)
    for frame in two_frames:
        case = cases[frame.sample.frame_type]
        for k in BUDGETS:
            prefix = frame.payload[:, :k]
            obs = case.plan.receive(case.layout.quantizer, prefix)
            result = receiver(case, backbone, frame.context, prefix)
            selected = result.index_select(1, case.layout.indices).unflatten(1, (4, 6)).movedim(2, -1)
            assert obs.mask_l1.sum() == k and not obs.mask_l2.any()
            assert torch.equal(selected[obs.mask_l1], obs.values[obs.mask_l1])
            assert torch.isfinite(result).all()
            assert not torch.equal(result, Receiver(SIGMA)(case, backbone, frame.context, prefix))


@torch.no_grad()
def test_t02_t11_fixed_models_receive_exact_independently_extracted_ids(native_run, native_frames, two_frames):
    backbone = native_run.codec.backbone
    receivers = [receiver_for(arm) for arm in ARM_NAMES]
    for frame in two_frames:
        case = native_frames[1][frame.sample.frame_type]
        encoded, _ = case.layout.encode(frame.sample)
        for k in BUDGETS[:-1]:
            independent = case.plan.encode_prefix(encoded, k)
            assert independent.shape == (1, k)
            assert torch.equal(independent, frame.payload[:, :k])
            for receive in receivers:
                prefix = independent.clone()
                receive(case, backbone, frame.context, prefix)
                assert torch.equal(prefix, frame.payload[:, :k])


@pytest.mark.parametrize("arm", ARM_NAMES)
@torch.no_grad()
def test_t04_intra_ignores_predictive_context(native_run, native_frames, two_frames, arm):
    intra, past = two_frames
    case = native_frames[1]["I"]
    receive = receiver_for(arm)
    receive.model.outputs["I"].bias.fill_(0.125)
    payload = intra.payload[:, :510]
    reference = receive(case, native_run.codec.backbone, intra.context, payload)
    poisoned = replace(intra.context, context=past.context.context, memory=past.context.memory)
    assert torch.equal(reference, receive(case, native_run.codec.backbone, poisoned, payload))


@pytest.mark.parametrize("arm", ARM_NAMES)
@torch.no_grad()
def test_t05_t09_t13_real_causal_reference_chain(native_run, native_frames, arm):
    clip, cases = native_frames
    codec = native_run.codec
    receiver = receiver_for(arm)
    for head in receiver.model.outputs.values():
        head.weight.normal_(std=0.01)
    pair = CausalPair()
    endpoints = set()
    for index, image in enumerate(clip):
        prior = pair.rx
        with pytest.raises(ValueError, match="consecutive"):
            codec.backbone.begin_frame(prior, index + 1)
        frame = pair.advance(codec, image, index, cases, receiver)
        assert same_state(pair.tx, pair.rx) and pair.rx.frame_id == index
        assert frame.context.intra == (index in (0, 32))
        if frame.context.intra:
            assert frame.context.context is None and frame.context.memory is None
        elif index == 1:
            case = cases["P"]
            payload = frame.payload[:, :510]
            other_context = replace(frame.context, context=torch.zeros_like(frame.context.context))
            assert not torch.equal(receiver(case, codec.backbone, frame.context, payload),
                                   receiver(case, codec.backbone, other_context, payload))
        if index + 1 in (2, 4, 16, 32):
            endpoints.add(index + 1)
            assert pair.rx.latent.shape == (1, 128, 68, 120)
            assert same_state(pair.tx, pair.rx)
    assert endpoints == {2, 4, 16, 32}
    with pytest.raises(RuntimeError, match="already committed"):
        frame.context.commit_reference_once(pair.rx.feature, pair.rx.latent)
    with pytest.raises(ValueError, match="consecutive"):
        pair.advance(codec, clip[-1], 32, cases, receiver)


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_t06_t07_t08_frozen_native_gradient_and_recovery_update(native_run, native_frames, two_frames, arm):
    backbone, projector = native_run.codec.backbone, native_run.codec.projector
    before = tuple(map(tensor_digest, (backbone, projector)))
    receiver = receiver_for(arm)
    model_before = tensor_digest(receiver.model)
    optimizer = torch.optim.AdamW(receiver.model.parameters(), lr=1e-3)
    for frame in two_frames:
        intra = frame.context.intra
        k = REF_BUDGET[intra]
        case = native_frames[1][frame.sample.frame_type]
        latent = receiver(case, backbone, frame.context, frame.payload[:, :k])
        latent.retain_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rgb, _ = backbone.decode_aligned(latent, frame.context, checkpointed=True)
        rgb_error = F.mse_loss(rgb.float(), frame.sample.rgb)
        grad, = torch.autograd.grad(rgb_error, latent, retain_graph=True)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        ref, _ = reference_supervision(backbone, projector, latent, frame.sample.native, frame.context)
        (rgb_error + 0.15 * ref).backward()
        assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
        grads = [p.grad for p in receiver.model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert tensor_digest(receiver.model) != model_before
    assert tuple(map(tensor_digest, (backbone, projector))) == before
    assert all(not p.requires_grad and p.grad is None for module in (backbone, projector) for p in module.parameters())


@torch.no_grad()
def test_t12_unreceived_level2_ids_are_invisible(native_run, two_frames):
    frame = two_frames[0]
    backbone = native_run.codec.backbone
    case = native_run.inputs.cases("SP-3", native_run.lock["keys"]["SP-3"])["I"]
    packet = case.encode(frame.sample)
    level, _ = case.plan.prefix(1024)
    first_refinement, = torch.nonzero(level, as_tuple=True)
    assert first_refinement.min() >= 510
    corrupt = packet.clone()
    corrupt[:, first_refinement] = (corrupt[:, first_refinement] + 1) % 64000
    assert torch.equal(corrupt[:, :510], packet[:, :510])
    assert not case.plan.receive(case.layout.quantizer, packet[:, :510]).mask_l2.any()
    for arm in ARM_NAMES:
        receive = receiver_for(arm)
        receive.model.outputs["I"].bias.fill_(0.125)
        original = receive(case, backbone, frame.context, packet[:, :510])
        assert torch.equal(original, receive(case, backbone, frame.context, corrupt[:, :510]))
        later = receive(case, backbone, frame.context, packet[:, :1024])
        assert not torch.equal(later, receive(case, backbone, frame.context, corrupt[:, :1024]))


@pytest.mark.parametrize("arm", ARM_NAMES)
@torch.no_grad()
def test_t15_full_native_1080p_rgb_inference(native_run, native_frames, two_frames, arm):
    backbone = native_run.codec.backbone
    receive = receiver_for(arm)
    for frame in two_frames:
        case = native_frames[1][frame.sample.frame_type]
        for k in (510, 8192):
            latent = receive(case, backbone, frame.context, frame.payload[:, :k])
            rgb, _ = backbone.decode_aligned(latent, frame.context)
            assert rgb.shape == (1, 3, 1080, 1920) and rgb.dtype == torch.float32
            assert torch.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
