"""Unified supervision must share observations while isolating causal transitions."""

import copy

import pytest
import torch
from torch import nn

from adaptok_hr.tests.test_training import settings
from adaptok_hr.training.losses import ReconstructionLoss
from adaptok_hr.training.train import JointTrainer, frame_objective


def unified_settings(probability=.5):
    result = settings()
    result.update(budget_mode="paired", budget_seed=42, paired_probability=probability)
    result["losses"] = {"l1": 1., "perceptual": 0., "vq": 1.}
    result["reconstruction"] = {"mode": "unified", "network": "alex", "global_size": [64, 96],
                                "crop_size": 32, "crop_seed": 77}
    return result


def test_unified_rejects_hand_assigned_supervision():
    for key in ("mse", "temporal", "monotonic", "detail", "suffix_detail", "chroma", "gan"):
        with pytest.raises(ValueError, match="unified"):
            ReconstructionLoss({"l1": 1., key: .1}, "cpu", {"mode": "unified"})


def test_paired_gradient_is_mean_of_independent_budgets_and_only_main_commits(small_model):
    model = small_model.train()
    frame = torch.rand(1, 3, 128, 256)
    config = unified_settings()
    losses = ReconstructionLoss(config["losses"], "cpu", config["reconstruction"])
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    total, state, row = frame_objective(model, frame, None, None, 4, 8, losses)
    handle.remove()
    assert len(calls) == 1
    total.backward()
    actual = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    first, expected, _ = frame_objective(model, frame, None, None, 4, None, losses)
    second, _, _ = frame_objective(model, frame, None, None, 8, None, losses)
    ((first + second) / 2).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(actual[name], parameter.grad, atol=2e-6, rtol=2e-4)
    assert torch.equal(state.rgb, expected.rgb)
    assert torch.equal(state.feature, expected.feature)
    assert row["reconstruction_loss"] == pytest.approx((row["budget_losses"]["4"] + row["budget_losses"]["8"]) / 2)
    assert state.rgb.grad_fn is not None


def test_paired_draws_match_across_frequencies_and_are_balanced(small_model):
    a, b = (JointTrainer(copy.deepcopy(small_model), unified_settings(p)) for p in (.5, .75))
    first = [a.budgets(1) for _ in range(12000)]
    second = [b.budgets(1) for _ in range(12000)]
    assert [r[0] for r in first] == [r[0] for r in second]
    assert .48 < sum(y is not None for _, y in first) / len(first) < .52
    assert .73 < sum(y is not None for _, y in second) / len(second) < .77
    assert all(aux is None or aux == second[i][1] for i, (_, aux) in enumerate(first))
    pairs = [tuple(sorted(r)) for r in first if r[1] is not None]
    for pair in ((4, 8), (8, 16), (16, 32)):
        assert .30 < pairs.count(pair) / len(pairs) < .36


def test_crop_views_use_identical_native_coordinates_and_independent_rng(monkeypatch):
    import lpips

    seen = []
    class PixelMetric(nn.Module):
        def forward(self, rgb, target):
            seen.append((rgb.detach().clone(), target.detach().clone()))
            return (rgb - target).square().mean()[None]
    monkeypatch.setattr(lpips, "LPIPS", lambda net: PixelMetric())
    config = unified_settings()["reconstruction"]
    losses = ReconstructionLoss({"l1": 1., "perceptual": .2}, "cpu", config)
    target = torch.arange(128 * 256).reshape(1, 1, 128, 256).expand(1, 3, -1, -1).float() / (128 * 256)
    rgb = (target * .5).requires_grad_()
    global_rng = torch.get_rng_state().clone()
    views = losses.sample_views(target)
    assert torch.equal(global_rng, torch.get_rng_state())
    value, terms = losses.distortion(rgb, target, views)
    x, y, w, h = views[0]
    assert w == h == 32
    torch.testing.assert_close(seen[1][0], rgb[..., y:y+h, x:x+w] * 2 - 1)
    torch.testing.assert_close(seen[1][1], target[..., y:y+h, x:x+w] * 2 - 1)
    assert seen[0][0].shape[-2:] == (64, 96)
    expected = (seen[0][0] - seen[0][1]).square().mean() / 2 + (seen[1][0] - seen[1][1]).square().mean() / 2
    torch.testing.assert_close(terms["perceptual"], expected)
    value.backward()
    assert rgb.grad.abs().sum() > 0
    seen.clear()
    losses.distortion(rgb, target, views)
    torch.testing.assert_close(seen[1][1], target[..., y:y+h, x:x+w] * 2 - 1)


def test_unified_crop_and_budget_rng_resume_exactly(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.training.checkpoint import restore_training, save_training

    config = {"model": small_model.config.to_dict(), "training": unified_settings()}
    trainer = JointTrainer(small_model, config["training"])
    sampler = ClipSampler([{"original_frames": 8}], 5)
    clip = torch.rand(2, 3, 128, 256)
    trainer.update(clip, 1)
    save_training(tmp_path / "one.pt", trainer, sampler, config, "native", 1, 2)
    expected_row = trainer.update(clip, 2)
    restored = JointTrainer(copy.deepcopy(small_model), config["training"])
    restore_training(tmp_path / "one.pt", restored, sampler, config, "native")
    row = restored.update(clip, 2)
    assert row["frame_metrics"] == expected_row["frame_metrics"]
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    saved = torch.load(tmp_path / "one.pt", weights_only=False)
    del saved["crop_rng"]
    torch.save(saved, tmp_path / "missing.pt")
    with pytest.raises((KeyError, ValueError)):
        restore_training(tmp_path / "missing.pt", restored, sampler, config, "native")


def test_cross_scale_shapes_gradients_positions_and_shared_initialization(small_config):
    from dataclasses import replace

    from adaptok_hr.models.model import AdapTokHR

    torch.manual_seed(9)
    original = AdapTokHR(small_config)
    torch.manual_seed(9)
    fused = AdapTokHR(replace(small_config, encoder_fusion="cross_scale_v1"))
    assert torch.equal(original.positions.positions, fused.positions.positions)
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, fused.state_dict()[name])
    x = torch.rand(1, 3, 128, 256)
    features = fused.encoder(x)
    assert [v.shape[-2:] for v in features] == list(small_config.grid_shapes)
    fused.forward_frame(x, None, 32).rgb.mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fused.encoder.cross_scale.parameters())
    assert "encoder_fusion" not in small_config.to_dict()
    assert fused.config.to_dict()["encoder_fusion"] == "cross_scale_v1"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("network", ["alex", "vgg"])
def test_native_unified_bf16_ip_uses_discrete_packets(network, record_property):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.models.model import AdapTokHR

    config = unified_settings(1.)
    config.update(amp_bf16=True)
    config["losses"]["perceptual"] = .1 if network == "alex" else .05
    config["reconstruction"].update(network=network, crop_size=256, global_size=[270, 480])
    model = AdapTokHR(ModelConfig(gradient_checkpointing=True, fine_input="local_residual",
                                 fine_decoder="masked_s32_v1", encoder_fusion="cross_scale_v1")).cuda()
    trainer = JointTrainer(model, config)
    row = trainer.update(torch.rand(2, 3, 1080, 1920), 1)
    assert [r["frame_type"] for r in row["frame_metrics"]] == ["I", "P"]
    assert all(r["monotonic_loss"] == r["temporal_loss"] == 0 for r in row["frame_metrics"])
    record_property(network + "_update_seconds", row["update_seconds"])
    record_property(network + "_peak_gpu_mib", row["peak_gpu_mib"])
    sender = Codec(model.eval(), True)
    receiver = Codec(copy.deepcopy(model).eval(), True)
    sending, receiving = None, None
    with torch.inference_mode():
        for _ in range(2):
            frame = torch.rand(1, 3, 1080, 1920, device="cuda")
            with sender.autocast():
                encoded = model.encode_features(frame, sending)
                for budget in model.config.budgets:
                    output = model.forward_frame(frame, sending, budget, encoded)
                    discrete, _ = model.decode_ids(output.ids, sending)
                    assert torch.equal(output.rgb, discrete)
            packet, sending = sender.encode_frame(frame, sending, 2048, "native-regression")
            rgb, receiving = receiver.decode_frame(packet, receiving)
            assert torch.equal(rgb, sending.rgb)
            assert torch.equal(receiving.feature, sending.feature)


@pytest.mark.gpu
@pytest.mark.slow
def test_native_fused_65_frame_independent_causal_loops(record_property):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from adaptok_hr.codec.bitstream import parse, truncate_packet
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.models.model import AdapTokHR

    model = AdapTokHR(ModelConfig(fine_input="local_residual", fine_decoder="masked_s32_v1",
                                 encoder_fusion="cross_scale_v1")).cuda().eval()
    sender, receiver = Codec(model, True), Codec(copy.deepcopy(model), True)
    frame = torch.rand(1, 3, 1080, 1920, device="cuda")
    with torch.inference_mode():
        for budget in model.config.budgets:
            sending, receiving = None, None
            for index in range(65):
                packet, sending = sender.encode_frame(frame, sending, budget, "long-native")
                rgb, receiving = receiver.decode_frame(packet, receiving)
                assert torch.equal(rgb, sending.rgb)
                assert torch.equal(receiving.feature, sending.feature)
                assert parse(packet).intra == (index % 32 == 0)
                if index % 32 == 0:
                    reset_rgb, _ = receiver.decode_frame(packet, None)
                    assert torch.equal(reset_rgb, rgb)
                if budget == 2048:
                    for prefix in (256, 512, 1024):
                        assert parse(truncate_packet(packet, prefix)).ids == parse(packet).ids[:prefix]
    record_property("native_long_loop_frames_per_budget", 65)
