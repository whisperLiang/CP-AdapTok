"""Check the decoder stage at its frame, loss and real packet boundaries."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F


def test_residual_decoder_start_and_training_preserve_absent_fine(small_model):
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.models.model import AdapTokHR

    parent = small_model.eval()
    child = AdapTokHR(replace(parent.config, fine_decoder=True)).eval()
    missing = child.load_state_dict(parent.state_dict(), strict=False)
    assert missing.missing_keys and all(k.startswith("decoder.fine_detail.") for k in missing.missing_keys)
    assert "fine_decoder" not in parent.config.to_dict()
    assert child.config.to_dict()["fine_decoder"] is True
    with pytest.raises(ValueError, match="fine_decoder"):
        replace(parent.config, fine_decoder="on")
    frame = torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        encoded = parent.encode_features(frame)
        for k in parent.config.budgets:
            expected = parent.forward_frame(frame, None, k, encoded)
            actual = child.forward_frame(frame, None, k, encoded)
            torch.testing.assert_close(actual.rgb, expected.rgb, rtol=0, atol=0)
            torch.testing.assert_close(actual.feature, expected.feature, rtol=0, atol=0)
    child.requires_grad_(False)
    branch = child.decoder.fine_detail.requires_grad_(True)
    opt = torch.optim.Adam(branch.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad()
        output = child.forward_frame(frame, None, 32, encoded)
        F.mse_loss(output.rgb, frame).backward()
        opt.step()
    low = parent.forward_frame(frame, None, 4, encoded)
    high = parent.forward_frame(frame, None, 32, encoded)
    torch.testing.assert_close(child.forward_frame(frame, None, 4, encoded).rgb, low.rgb, atol=0, rtol=0)
    assert not torch.equal(child.forward_frame(frame, None, 32, encoded).rgb, high.rgb)
    sender, receiver = Codec(child), Codec(copy.deepcopy(child))
    packet, state = sender.encode_frame(frame, None, 32)
    rgb, decoded = receiver.decode_frame(packet, None)
    torch.testing.assert_close(rgb, state.rgb, atol=0, rtol=0)
    torch.testing.assert_close(decoded.feature, state.feature, atol=0, rtol=0)
    with pytest.raises(ValueError, match="identity"):
        Codec(parent).decode_frame(packet)


def test_highpass_loss_matches_reflected_rgb_and_nested_weights(small_model):
    from adaptok_hr.training.losses import ReconstructionLoss
    from adaptok_hr.training.train import frame_objective

    def reference(x):
        return x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)

    frame = torch.rand(1, 3, 128, 256)
    reconstruct = ReconstructionLoss({"detail": .1}, "cpu")
    shifted = frame + .15
    nearly_zero, _ = reconstruct.distortion(shifted, frame)
    assert nearly_zero.item() < 1e-6
    rgb = torch.rand_like(frame, requires_grad=True)
    metric = F.l1_loss(reference(rgb), reference(frame))
    value, terms = reconstruct.distortion(rgb, frame)
    torch.testing.assert_close(terms["detail"], metric)
    torch.testing.assert_close(value, .1 * metric)
    actual_gradient = torch.autograd.grad(value, rgb)[0]
    expected_gradient = torch.autograd.grad(.1 * metric, rgb)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    encoded = small_model.encode_features(frame)
    with torch.no_grad():
        hi = small_model.forward_frame(frame, None, 32, encoded)
        lo = small_model.forward_frame(frame, None, 16, encoded)
        expected = .05 * (F.l1_loss(reference(hi.rgb), reference(frame)) +
                          F.l1_loss(reference(lo.rgb), reference(frame)))
    loss, state, _ = frame_objective(small_model, frame, None, None, 16, 32, reconstruct)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(state.rgb, lo.rgb, atol=0, rtol=0)


@pytest.mark.gpu
def test_native_decoder_detail_training_updates_and_decodes_real_packet():
    if not torch.cuda.is_available():
        pytest.skip("native CUDA integration")
    from adaptok_hr.codec.codec import Codec
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.data.dataset import read_clip
    from adaptok_hr.data.manifest import load_index
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.runtime import read_config, seed_everything
    from adaptok_hr.training.train import JointTrainer

    seed_everything(17)
    config = read_config("experiments/adaptok_hr/fine_recovery/configs/A1.yaml")
    config["model"].update(fine_decoder=True, fine_input="local_residual")
    config["training"].update(budget_mode="fixed", fixed_budget=2048)
    config["training"]["losses"].update(perceptual=0., detail=.1, suffix_detail=.25)
    model = AdapTokHR(ModelConfig(**config["model"])).cuda()
    trainer = JointTrainer(model, config["training"])
    src = load_index(config["data"]["index"])["splits"]["train"][0]
    frames, _ = read_clip(src, 0, 2)
    before = {n: x.detach().clone() for n, x in model.decoder.fine_detail.named_parameters()}
    stats = trainer.update(frames, 1)
    assert all(float(r["detail"]) > 0 for r in stats["frame_metrics"])
    assert any(not torch.equal(before[n], x) for n, x in model.decoder.fine_detail.named_parameters())
    sender, receiver = Codec(model.eval(), True), Codec(copy.deepcopy(model).eval(), True)
    sending = receiving = None
    for frame in frames:
        data, sending = sender.encode_frame(frame[None].cuda(), sending, 2048)
        assert len(data) == 3470
        _, receiving = receiver.decode_frame(data, receiving)
        assert torch.equal(sending.rgb, receiving.rgb) and torch.equal(sending.feature, receiving.feature)
