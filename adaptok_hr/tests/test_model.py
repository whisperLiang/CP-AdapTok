import pytest
import torch
from torch.nn import functional as F


def test_native_positions_and_distributed_prefix():
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.models.tokenizer import PositionTable

    table = PositionTable(ModelConfig())
    assert table.shapes == ((9, 15), (17, 30), (34, 60))
    assert tuple(table.positions.shape) == (2685, 3)
    assert table.positions[:135, 0].eq(0).all()
    assert table.positions[135:645, 0].eq(1).all()
    assert len(set(map(tuple, table.positions.tolist()))) == 2685
    for start, shape in zip((0, 135, 645), table.shapes):
        points = table.positions[start:start + min(16, shape[0] * shape[1]), 1:]
        assert points[:, 0].min() < shape[0] / 3
        assert points[:, 0].max() >= shape[0] * 2 / 3
        assert points[:, 1].min() < shape[1] / 3
        assert points[:, 1].max() >= shape[1] * 2 / 3


def test_window_padding_and_actual_local_computation(monkeypatch):
    from adaptok_hr.models.window_attention import WindowAttention

    layer = WindowAttention(12, heads=3, window=4).eval()
    original = F.scaled_dot_product_attention
    calls = []

    def observe(q, k, v, **kwargs):
        calls.append((q.shape, kwargs["attn_mask"]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    x = torch.randn(1, 12, 5, 7)
    y = layer(x)
    altered = x.clone()
    altered[:, :, 4:, 4:] += 100
    z = layer(altered)
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, :, :4, :4], z[:, :, :4, :4], rtol=0, atol=0)
    assert calls and all(shape[-2] == 16 for shape, _ in calls)
    assert calls[0][0][0] == 4  # four independent windows, never a 35x35 matrix
    assert calls[0][1].dtype == torch.bool
    assert calls[0][1].sum().item() == 35
    assert torch.isfinite(y).all()


def test_window_matches_separate_unpadded_edge_attention():
    from adaptok_hr.models.window_attention import WindowAttention

    layer = WindowAttention(12, heads=3, window=4).eval()
    x = torch.randn(1, 12, 5, 7)
    # The bottom-right partial window has only 1x3 real keys.
    full = layer(x)[:, :, 4:, 4:]
    isolated = layer(x[:, :, 4:, 4:])
    torch.testing.assert_close(full, isolated, atol=1e-6, rtol=1e-5)


def test_vq_discrete_forward_gradients_and_range():
    from adaptok_hr.models.vq import VectorQuantizer

    vq = VectorQuantizer(8192, 16)
    x = torch.randn(1, 42, 16, requires_grad=True)
    out = vq(x)
    assert out.ids.min() >= 0 and out.ids.max() < 8192
    torch.testing.assert_close(out.quantized, vq.lookup(out.ids), rtol=0, atol=0)
    (out.quantized.square().mean() + out.loss).backward()
    assert x.grad.abs().sum() > 0 and vq.codebook.weight.grad.abs().sum() > 0
    assert out.stats["perplexity"] > 0
    assert torch.isfinite(x.grad).all()


def test_nested_prefix_and_only_transmitted_values(small_model):
    model = small_model.eval()
    x = torch.rand(1, 3, 128, 256)
    encoded = model.encode_features(x, None)
    for budget in model.config.budgets:
        direct = model.token_ids(x, None, budget)
        assert torch.equal(encoded.ids[:, :budget], direct)
        rgb, _ = model.decode_ids(direct, None)
        assert rgb.shape == x.shape
        changed = encoded.quantized.detach().clone()
        changed[:, budget:] = float("nan")
        a, _ = model.decoder(encoded.quantized[:, :budget], None)
        b, _ = model.decoder(changed[:, :budget], None)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(rgb, a, rtol=0, atol=0)


def test_scatter_places_received_values_and_marks_absence(small_model):
    table = small_model.positions
    tokens = torch.arange(8 * 16).float().reshape(1, 8, 16)
    maps = table.scatter(tokens)
    for level, ((h, w), capacity) in enumerate(zip(table.shapes, table.counts)):
        offset = sum(table.counts[:level])
        sent = max(0, min(8 - offset, capacity))
        mask = maps[level][:, -1].reshape(-1)
        assert int(mask.sum()) == sent
        for i in range(sent):
            flat = int(table.order(level)[i])
            torch.testing.assert_close(maps[level][0, :-1, flat // w, flat % w], tokens[0, offset + i])
            assert mask[flat] == 1
        missing = maps[level][:, :-1].flatten(2)[:, :, mask == 0]
        assert missing.eq(0).all()


def test_past_is_independent_of_future_and_gop_reset(small_model):
    model = small_model.eval()
    clip = torch.rand(1, 3, 3, 128, 256)
    a = model(clip, budget=16)
    changed = clip.clone()
    changed[:, :, 2] = 1 - changed[:, :, 2]
    b = model(changed, budget=16)
    assert torch.equal(a[0].ids, b[0].ids) and torch.equal(a[1].ids, b[1].ids)
    torch.testing.assert_close(a[1].rgb, b[1].rgb, rtol=0, atol=0)
    from adaptok_hr.models.temporal_state import TemporalState

    prior = TemporalState(stream_id=b"a" * 16, frame_index=31, rgb=torch.rand_like(a[0].rgb),
                          feature=a[0].feature, chain=b"b" * 32)
    reset = model.forward_frame(clip[:, :, 0], prior, 16)
    first = model.forward_frame(clip[:, :, 0], None, 16)
    assert reset.frame_type == "I" and reset.state.frame_index == 32
    torch.testing.assert_close(reset.rgb, first.rgb, rtol=0, atol=0)
    assert torch.equal(reset.ids, first.ids)


def test_forward_backward_updates_all_modules(small_model):
    model = small_model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model.forward_frame(torch.rand(1, 3, 128, 256), None, 32)
    loss = result.rgb.square().mean() + result.vq_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    for prefix in ("encoder.", "tokenizer.", "vq.codebook.", "decoder."):
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith(prefix))


@pytest.mark.gpu
@pytest.mark.slow
def test_real_native_1080p_forward_backward_and_crop():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.models.model import AdapTokHR

    model = AdapTokHR(ModelConfig(gradient_checkpointing=True)).cuda().train()
    x = torch.rand(1, 3, 1080, 1920, device="cuda")
    padded = model.encoder.pad_input(x)
    assert padded.shape[-2:] == (1088, 1920)
    torch.testing.assert_close(padded[:, :, :1080], x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = model.forward_frame(x, None, 2048)
        second = model.forward_frame(x, first.state, 256)
        assert first.rgb.shape == second.rgb.shape == x.shape
        assert second.frame_type == "P"
        loss = F.mse_loss(second.rgb.float(), x) + first.vq_loss + second.vq_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()


@pytest.mark.gpu
@pytest.mark.slow
def test_native_bf16_tbptt4_exact_optimizer_resume(tmp_path, record_property):
    from pathlib import Path

    if not torch.cuda.is_available() or not Path("data/adaptok_hr/index.json").exists():
        pytest.skip("CUDA and locally indexed native Train source required")
    from adaptok_hr.config import ModelConfig
    from adaptok_hr.data.dataset import ClipSampler, read_clip
    from adaptok_hr.data.manifest import load_index
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.runtime import read_config, seed_everything
    from adaptok_hr.training.checkpoint import restore_training, save_training
    from adaptok_hr.training.train import JointTrainer

    config = read_config("configs/adaptok_hr/default.yaml")
    config["training"]["tbptt"] = 4
    config["training"]["losses"]["perceptual"] = 0.0
    seed_everything(123)
    index = load_index("data/adaptok_hr/index.json")
    source = index["splits"]["train"][0]
    clip, _ = read_clip(source, 0, 4)
    trainer = JointTrainer(AdapTokHR(ModelConfig(**config["model"])).cuda(), config["training"])
    sampler = ClipSampler([source], 42)
    first = trainer.update(clip, 1)
    record_property("tbptt4_peak_gpu_mib", first["peak_gpu_mib"])
    save_training(tmp_path / "native.pt", trainer, sampler, config, index["identity"], 1, 4)
    expected_row = trainer.update(clip, 2)
    expected = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    restored = JointTrainer(AdapTokHR(ModelConfig(**config["model"])).cuda(), config["training"])
    restore_training(tmp_path / "native.pt", restored, sampler, config, index["identity"])
    actual_row = restored.update(clip, 2)
    for key, value in expected.items():
        torch.testing.assert_close(restored.model.state_dict()[key].cpu(), value, rtol=0, atol=0)
    assert actual_row["budgets"] == expected_row["budgets"]
