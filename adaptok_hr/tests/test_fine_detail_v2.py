from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F


def test_spatial_residual_uses_neighbors_and_does_not_change_coarse_tokens(small_config):
    from adaptok_hr.models.tokenizer import PositionTable, Tokenizer

    table = PositionTable(small_config)
    direct = Tokenizer(small_config, table)
    residual = Tokenizer(replace(small_config, fine_input="local_residual"), table)
    residual.load_state_dict(direct.state_dict())
    features = [torch.randn(1, c, h, w) for c, (h, w) in zip(reversed(small_config.channels[4:]), table.shapes)]
    fine = residual.projections[-1](features[-1]).float()
    padded = F.pad(fine, (1, 1, 1, 1), mode="replicate")
    centered = fine - F.avg_pool2d(padded, 3, 1)
    expected = F.normalize(centered.flatten(2).transpose(1, 2)[:, table.order(2)], dim=-1)
    offset = sum(table.counts[:2])
    actual = residual(features)
    torch.testing.assert_close(actual[:, offset:], expected)
    torch.testing.assert_close(actual[:, :offset], direct(features)[:, :offset], atol=0, rtol=0)
    flat = [v.detach().fill_(1).requires_grad_() for v in features]
    token = residual(flat)[:, offset:]
    unit = torch.zeros(16)
    unit[0] = 1
    torch.testing.assert_close(token, unit.expand_as(token))
    token.sum().backward()
    assert torch.isfinite(flat[-1].grad).all()


def test_suffix_targets_use_transmission_order_and_valid_native_pixels():
    from adaptok_hr.training.fine_detail import detail_targets

    # A 32x32 cell spans sixteen 8x8 bins. The last 8 rows of cell 3 are padding.
    image = torch.zeros(1, 3, 56, 64)
    numbers = torch.arange(12).float().reshape(3, 4).repeat_interleave(8, 0).repeat_interleave(8, 1)
    image[:, 0, 32:, 32:] = numbers
    target, valid = detail_targets(image, torch.zeros_like(image), (2, 2), torch.tensor([3, 0]))
    torch.testing.assert_close(target[0, 0, :12], (torch.arange(12) - 5.5) * .2126)
    assert valid[0, 0].tolist() == [True] * 12 + [False] * 4
    assert target[0, 0, 12:].eq(0).all() and target[0, 1].eq(0).all()
    target, _ = detail_targets(image, image, (2, 2), torch.tensor([3]))
    assert target.eq(0).all()


@pytest.mark.parametrize("budget,aux,weight", [(32, None, 1.), (16, 32, .5), (32, 16, .5), (4, 32, .5), (16, 8, 0.)])
def test_suffix_objective_targets_only_sent_extra_ids_and_commits_main(small_model, budget, aux, weight):
    from adaptok_hr.training.fine_detail import detail_targets, suffix_angular_loss
    from adaptok_hr.training.losses import ReconstructionLoss
    from adaptok_hr.training.train import frame_objective

    model = small_model
    frame = torch.rand(1, 3, 128, 256)
    encoded = model.encode_features(frame)
    encoded.quantized.retain_grad()
    model.encode_features = lambda *_: encoded
    with torch.no_grad():
        coarse = model.forward_frame(frame, None, 16, encoded).rgb
        expected = model.forward_frame(frame, None, budget, encoded)
    loss, state, row = frame_objective(model, frame, None, None, budget, aux,
                                       ReconstructionLoss({"suffix_detail": .25}, "cpu"))
    positions = model.positions.positions[16:32]
    index = positions[:, 1] * 8 + positions[:, 2]
    target, valid = detail_targets(frame, coarse, (4, 8), index)
    expected_loss = suffix_angular_loss(encoded.quantized[:, 16:32], target, valid) * .25 * weight
    torch.testing.assert_close(loss, expected_loss)
    assert row["suffix_detail_loss"] == pytest.approx(float(expected_loss.detach()))
    torch.testing.assert_close(state.rgb, expected.rgb, atol=0, rtol=0)
    loss.backward()
    grad = encoded.quantized.grad
    assert grad[:, :16].eq(0).all() and grad[:, 32:].eq(0).all()
    assert bool(grad[:, 16:32].abs().sum() > 0) == bool(weight)
    assert all(p.grad is None or p.grad.eq(0).all() for p in model.decoder.parameters())


def test_angular_loss_ignores_padding_flat_regions_and_matches_hand_value():
    from adaptok_hr.training.fine_detail import suffix_angular_loss

    values = torch.zeros(1, 2, 16, requires_grad=True)
    target = torch.zeros_like(values)
    target[0, 0, :2] = torch.tensor([1., -1.])
    valid = torch.zeros_like(values, dtype=torch.bool)
    valid[..., :2] = True
    with torch.no_grad():
        values[0, 0, :2] = torch.tensor([-1., 1.])
        values[..., 2:] = 100.
    loss = suffix_angular_loss(values, target, valid)
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert values.grad[:, 1].eq(0).all() and values.grad[:, :, 2:].eq(0).all()


def test_legacy_codec_config_identity_and_new_mode_rejection(small_model):
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.models.model import AdapTokHR

    frame = torch.rand(1, 3, 128, 256)
    old = Codec(small_model.eval())
    packet, _ = old.encode_frame(frame, budget=32)
    assert "fine_input" not in small_model.config.to_dict()
    changed = AdapTokHR(replace(small_model.config, fine_input="local_residual")).eval()
    changed.load_state_dict(small_model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        Codec(changed).decode_frame(packet)
    with pytest.raises(ValueError, match="fine_input"):
        replace(small_model.config, fine_input="unknown")


@pytest.mark.gpu
def test_native_combined_objective_resumes_exactly_and_reads_archived_packet(tmp_path):
    from pathlib import Path

    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.data.dataset import ClipSampler, read_clip
    from adaptok_hr.data.manifest import load_index
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.runtime import read_config, seed_everything
    from adaptok_hr.training.checkpoint import load_model, restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    if not torch.cuda.is_available():
        pytest.skip("native CUDA integration")
    seed_everything(23)
    config = read_config("experiments/adaptok_hr/fine_recovery/configs/A1.yaml")
    index = load_index(config["data"]["index"])
    name = "DAVIS2017:car-roundabout"
    source = next(row for row in index["splits"]["validation"] if row["content_id"] == name)
    frame, _ = read_clip(source, 0, 1)
    old_path = Path("checkpoints/adaptok_hr_fine_recovery/A1/step_001000.pt")
    packet_path = Path("experiments/adaptok_hr/fine_recovery/evidence/A1/validation_001000_64f/same_context_packets/DAVIS2017_car-roundabout/frame_0000_K2048.bin")
    if old_path.is_file():
        old, _ = load_model(old_path)
        codec = Codec(old, amp_bf16=True)
        expected = packet_path.read_bytes()
        actual, sender = codec.encode_frame(frame.cuda(), None, 2048, name)
        assert actual == expected
        rgb, _ = codec.decode_frame(expected)
        assert torch.equal(rgb, sender.rgb)
    source = index["splits"]["train"][0]
    clip, _ = read_clip(source, 0, 2)
    config["model"]["fine_input"] = "local_residual"
    config["training"].update(budget_mode="fixed", fixed_budget=2048)
    config["training"]["losses"].update(perceptual=0., suffix_detail=.25)
    model = AdapTokHR(ModelConfig(**config["model"])).cuda()
    trainer = JointTrainer(model, config["training"])
    sampler = ClipSampler([source], 1)
    first = trainer.update(clip, 1)
    assert all(row["suffix_detail_loss"] > 0 for row in first["frame_metrics"])
    path = tmp_path / "native.pt"
    save_training(path, trainer, sampler, config, index["identity"], 1, 2)
    expected = trainer.update(clip, 2)
    resumed = JointTrainer(AdapTokHR(model.config).cuda(), config["training"])
    restore_training(path, resumed, sampler, config, index["identity"])
    actual = resumed.update(clip, 2)
    assert actual["mse"] == expected["mse"]
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
