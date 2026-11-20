import math

import pytest
import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.joint.loss import JointLoss
from aip_vtok.joint.protocol import CONFIG, open_experiment
from aip_vtok.joint.scan import GROUPS, SLOT_COUNT
from aip_vtok.joint.trainer import BudgetSampler, History, Trainer, Window, window_for

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


def test_window_and_budget_resume_sequences():
    rows = {"test-train-video": {"original_frames": 80}, "second": {"original_frames": 27}}
    for i in range(1, 102):
        window = window_for(rows, 1, i, 8)
        assert window == window_for(rows, 1, i, 8)
        assert window.anchor % 32 == 0
        assert window.start + window.length <= rows[window.video]["original_frames"]
    sampler = BudgetSampler(5)
    for i in range(13):
        sampler.draw(i % 3 == 0)
    clone = BudgetSampler(5)
    clone.load_state_dict(sampler.state_dict())
    a, b = [], []
    for i in range(100):
        a.append(sampler.draw(i % 3 == 0))
        b.append(clone.draw(i % 3 == 0))
    assert a == b and set(k for row in a for k in row) == {510, 1024, 2048, 4096, 8192, 16320, 32640, 65280}


@pytest.mark.gpu
@GPU
def test_native_rgb_dists_and_rate_gradients():
    run = open_experiment(CONFIG, 1)
    codec = run.codec
    clip = run.data.read("train", "DAVIS2017:bear", 0, 2)
    first, second = [x[None].cuda().float() / 255 for x in clip]
    backbone, table = codec.backbone, codec.scan.table()
    context = backbone.begin_frame(ReferenceState(), 0)
    with torch.no_grad():
        target = backbone.aligned_latent(first, context)
    assert target.shape == (1, 128, 68, 120)
    matrix = codec.transform.matrix(True)
    fsq = codec.fsq.encode_train(codec.coefficients(target, matrix), True)
    mask = codec.scan.mask(table, 2048)
    latent = codec.recover(fsq.values.reshape(1, SLOT_COUNT, 6) * mask[None, :, None],
                           mask, context, 2048, matrix, checkpointed=True)
    rec, _ = backbone.decode_aligned(latent, context, checkpointed=True)
    assert rec.shape == first.shape
    parameters = (codec.transform.skew, codec.fsq.log_scale, codec.scan.score, codec.sld.inputs["I"].weight)
    for grad in torch.autograd.grad(F.mse_loss(rec, first), parameters, retain_graph=True):
        assert grad.isfinite().all() and grad.abs().sum() > 0
    bits = codec.entropy.bits(fsq.ids.reshape(1, SLOT_COUNT), codec.slot_groups, True,
                              fsq.digit_position.reshape(1, SLOT_COUNT, -1))
    for grad in torch.autograd.grad((bits * mask).mean(), (codec.fsq.log_scale, codec.entropy.logits)):
        assert grad.isfinite().all() and grad.abs().sum() > 0
    objective = JointLoss(run.config["loss"]).cuda().to(memory_format=torch.channels_last)
    crop = rec.detach()[:, :, :256, :256].requires_grad_()
    _, dists = objective.perceptual(first[:, :, :256, :256], crop)
    grad, = torch.autograd.grad(dists, crop)
    assert grad.isfinite().all() and grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in objective.parameters())
    with torch.no_grad():
        hard = codec.recover_ids(fsq.ids.reshape(1, -1)[:, table[:2048]], table, context, matrix)
        assert torch.equal(latent, hard)
        reference = context.commit_reference_once(codec.projector(hard, True), hard)
        p = backbone.begin_frame(reference, 1)
        assert backbone.aligned_latent(second, p).shape == target.shape
    assert codec.slot_groups.unique().numel() == GROUPS
    run.verify_integrity()
    assert run.data.counts["test"] == 0


@pytest.mark.gpu
@GPU
def test_real_native_bf16_update():
    run = open_experiment(CONFIG, 1)
    trainer = Trainer(run.codec, run.config, 1)
    window = Window("DAVIS2017:bear", 0, 2)
    clip = run.data.read("train", window.video, window.anchor, window.span)
    before = run.codec.sld.inputs["I"].weight.detach().clone()
    result = trainer.update(clip, window, 1)
    assert math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0
    assert not torch.equal(before, run.codec.sld.inputs["I"].weight)
    assert trainer.scheduler.last_epoch == 1
    assert set(f["frame_type"] for f in result["frames"]) == {"I", "P"}
    assert all(len(f["budgets"]) >= 2 for f in result["frames"])
    run.verify_integrity()


def test_history_mismatch_is_fatal():
    history = History.at(32)
    left = ReferenceState(frame_id=32, latent=torch.zeros(1))
    right = ReferenceState(frame_id=32, latent=torch.ones(1))
    with pytest.raises(RuntimeError, match="differ"):
        history.submit(left, right)
    assert history.rx.frame_id == 31


def test_schedule_phases_export_one_matching_frozen_scan(tmp_path):
    import yaml

    from aip_vtok.joint.codec import JointCodec
    from aip_vtok.joint.protocol import initial_coordinates
    from aip_vtok.joint.trainer import export_scan

    config = yaml.safe_load(open(CONFIG))
    orders, levels, scales = initial_coordinates()
    codec = JointCodec(torch.nn.Identity(), torch.nn.Identity(), orders, levels, scales, torch.arange(SLOT_COUNT))
    trainer = Trainer(codec, config, 1, objective=torch.nn.Identity())
    assert trainer.schedule(2000) == "A"
    assert not codec.transform.skew.requires_grad and not codec.scan.score.requires_grad
    assert trainer.schedule(2001) == "B"
    assert codec.transform.skew.requires_grad and codec.scan.score.requires_grad and codec.entropy.logits.requires_grad
    table = codec.scan.table()
    export_scan(codec.scan, tmp_path)
    assert trainer.schedule(24001) == "B"
    assert trainer.schedule(28001) == "C"
    assert torch.equal(table, codec.scan.table())
    assert not codec.scan.score.requires_grad
    assert (tmp_path / "scan_v1.u16le").stat().st_size == 2 * SLOT_COUNT
