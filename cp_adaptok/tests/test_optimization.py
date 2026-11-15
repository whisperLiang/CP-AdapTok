import copy

import pytest
import torch

from cp_adaptok.diagnostics import distribution_stats
from cp_adaptok.tokenizer.vector_quantizer import AdapTokVQBottleneck
from cp_adaptok.training import set_quantization_step, teacher_applies, temperature_at


def test_distribution_distinguishes_token_collapse_from_layer_norm_variance():
    repeated = torch.randn(1, 1, 16).expand(1, 128, 16)
    stats = distribution_stats(repeated)
    assert stats["std"] > 0
    assert stats["token_variance"] < 1e-12
    assert stats["effective_rank"] == 0
    assert stats["query_cosine_mean"] == pytest.approx(1)
    varied = distribution_stats(torch.randn(1, 128, 16))
    assert varied["effective_rank"] > 8


def test_stochastic_training_never_changes_deterministic_inference():
    torch.manual_seed(7)
    vq = AdapTokVQBottleneck(16, 32, 8)
    vq.set_training_policy("stochastic_training", temperature=0.3)
    features = torch.randn(1, 128, 16, requires_grad=True)
    first, second = vq(features), vq(features)
    assert not torch.equal(first.ids, second.ids)
    (first.embeddings.square().mean() + first.loss).backward()
    assert features.grad.norm() > 0
    assert vq.codebook.weight.grad.norm() > 0
    vq.eval()
    expected = vq(features)
    vq.set_training_policy("deterministic_nearest")
    assert torch.equal(expected.ids, vq(features).ids)
    torch.testing.assert_close(expected.embeddings, vq.embed(expected.ids))


def test_continuous_is_a_training_diagnostic_and_has_no_stream_ids():
    vq = AdapTokVQBottleneck(16, 32, 8)
    vq.set_training_policy("continuous")
    x = torch.randn(1, 8, 16)
    result = vq(x)
    assert result.ids is None
    torch.testing.assert_close(result.embeddings, vq.project_out(vq.project_features(x)))
    assert result.loss == 0
    with pytest.raises(RuntimeError, match="continuous"):
        vq.eval()(x)


def test_kmeans_requires_enough_distinct_real_samples_and_is_reproducible():
    vq = AdapTokVQBottleneck(16, 16, 8)
    other = copy.deepcopy(vq)
    samples = torch.nn.functional.normalize(torch.randn(128, 8), dim=-1)
    vq.initialize_codebook(samples, seed=13, iterations=4)
    other.initialize_codebook(samples, seed=13, iterations=4)
    assert torch.equal(vq.codebook.weight, other.codebook.weight)
    assert vq.codebook.weight.unique(dim=0).shape[0] == 16
    with pytest.raises(ValueError, match="distinct"):
        vq.initialize_codebook(samples[:1].expand(128, -1), seed=13)


def test_teacher_default_never_supervises_mismatched_p_history():
    assert teacher_applies({"policy": "i_only"}, "I")
    assert not teacher_applies({"policy": "i_only"}, "P")
    assert not teacher_applies({"policy": "none"}, "I")
    with pytest.raises(ValueError, match="diagnostic"):
        teacher_applies({"policy": "legacy_ip"}, "P")
    assert teacher_applies({"policy": "legacy_ip", "allow_mismatched_p_diagnostic": True}, "P")
    with pytest.raises(ValueError, match="verified"):
        teacher_applies({"policy": "matched_p"}, "P")


def test_closed_loop_training_never_requests_native_p_target(codec):
    from cp_adaptok.training import configure_stage, train_clip

    class IntraTeacher:
        last_provenance = {"latent_source": "exact-test-intra"}

        def reset(self):
            self.calls = 0

        def target(self, rgb):
            self.calls += 1
            assert self.calls == 1
            return rgb.new_zeros((1, 256, 4, 4))

    model = copy.deepcopy(codec).train()
    config = {
        "stage": 2,
        "training": {"tbptt": 2, "amp_bf16": False},
        "loss": {"mse": 1.0, "vq": 0.1, "latent": 1.0, "estimated_rate": 0.0, "monotonic": 0.1},
        "teacher": {"policy": "i_only"},
    }
    configure_stage(model, 2, config)
    teacher = IntraTeacher()
    records = train_clip(model, torch.rand(4, 3, 16, 24), [8, 32], config, {}, teacher, 1.0)
    assert teacher.calls == 1
    assert records[0]["teacher"]["reference_condition"] == "intra_no_history"
    assert all(row["teacher"] is None for row in records[1:])


def test_disabled_teacher_and_auxiliary_training_do_not_require_native_extensions(monkeypatch):
    import importlib.util

    from cp_adaptok.runtime import preflight

    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    for settings in [{"stage": 2, "teacher": {"policy": "none"}}, {"stage": 3}]:
        preflight(dict(settings, loss={"latent": 1.0}, paths={}, data={}), "train", allow_untrained=True)


def test_temperature_schedule_is_absolute_step_based_for_resume():
    config = {"temperature": 1.0, "final_temperature": 0.1, "anneal_steps": 100}
    assert temperature_at(config, 0) == 1
    assert temperature_at(config, 100) == pytest.approx(0.1)
    assert temperature_at(config, 300) == pytest.approx(0.1)
    assert 0.1 < temperature_at(config, 50) < 1


def test_scorer_and_entropy_train_on_actual_deterministic_tokens(codec):
    model = copy.deepcopy(codec).train()
    config = {"stage": 3, "training": {"vq": {"strategy": "stochastic_training", "warmup_steps": 20}}}
    set_quantization_step(model, config, 0)
    assert model.vq.strategy == "deterministic_nearest"
    features = torch.randn(1, 32, 24)
    assert torch.equal(model.vq(features).ids, model.vq(features).ids)


def test_full_codebook_sampling_matches_adaptok_cosine_softmax_distribution():
    torch.manual_seed(41)
    vq = AdapTokVQBottleneck(2, 2, 2)
    with torch.no_grad():
        vq.project_in.weight.copy_(torch.eye(2))
        vq.project_in.bias.zero_()
        vq.codebook.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    vq.set_training_policy("stochastic_training", temperature=1.0)
    ids = vq(torch.tensor([[[1.0, 0.0]]]).expand(1, 12000, 2)).ids
    actual = (ids == 0).float().mean().item()
    expected = torch.tensor([1.0, -1.0]).softmax(0)[0].item()
    assert abs(actual - expected) < 0.015


def test_query_initialization_rescales_the_learned_residual_without_spatial_assignment(codec):
    model = copy.deepcopy(codec).train()
    before = model.tokenizer.queries.detach().clone()
    model.tokenizer.rescale_initial_queries(50.0)
    torch.testing.assert_close(model.tokenizer.queries, before * 50, rtol=0, atol=0)
    assert model.tokenizer.queries.shape == before.shape


def test_dead_codes_wait_for_a_window_and_clear_only_replaced_optimizer_rows():
    torch.manual_seed(23)
    vq = AdapTokVQBottleneck(16, 32, 8)
    vq.configure_dead_codes(window=3, max_replacements=4)
    opt = torch.optim.AdamW(vq.parameters(), lr=0.001)
    for update in range(3):
        opt.zero_grad()
        result = vq(torch.ones(1, 8, 16))
        result.loss.backward()
        opt.step()
        before = opt.state[vq.codebook.weight]["exp_avg"].clone()
        changed = vq.finish_update(opt)
        if update < 2:
            assert not changed
        else:
            assert len(changed) == 4
            assert opt.state[vq.codebook.weight]["exp_avg"][changed].count_nonzero() == 0
            retained = [i for i in range(32) if i not in changed]
            assert torch.equal(opt.state[vq.codebook.weight]["exp_avg"][retained], before[retained])


def test_stochastic_vq_checkpoint_restores_sampling_optimizer_and_window(codec, tmp_path):
    from cp_adaptok.runtime import load_checkpoint, save_checkpoint
    from cp_adaptok.training import configure_stage, restore_training

    model = copy.deepcopy(codec).train()
    config = {"stage": 1}
    configure_stage(model, 1, config)
    model.vq.set_training_policy("stochastic_training", 0.2)
    model.vq.configure_dead_codes(window=2, max_replacements=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(1, 32, 24)

    def update():
        opt.zero_grad()
        result = model.vq(x)
        (result.embeddings.square().mean() + result.loss).backward()
        opt.step()
        replaced = model.vq.finish_update(opt)
        return result.ids.clone(), replaced

    update()
    save_checkpoint(tmp_path / "vq.pt", model, opt, rng, 1, config, "test")
    artifact = load_checkpoint(tmp_path / "vq.pt")
    expected, changed = update()
    weights = model.vq.codebook.weight.detach().clone()
    moments = opt.state[model.vq.codebook.weight]["exp_avg"].clone()
    model.load_state_dict(artifact["model"])
    restore_training(model, opt, rng, artifact, 1)
    actual, resumed_changed = update()
    assert torch.equal(actual, expected)
    assert changed == resumed_changed
    assert torch.equal(model.vq.codebook.weight, weights)
    assert torch.equal(opt.state[model.vq.codebook.weight]["exp_avg"], moments)


def test_native_image_sequence_rectangular_crop_and_content_isolation(tmp_path):
    import cv2
    import numpy as np

    from cp_adaptok.data import check_splits, read_video

    folder = tmp_path / "source"
    folder.mkdir()
    for i in range(4):
        cv2.imwrite(str(folder / f"{i:05d}.png"), np.full((80, 120, 3), i * 30, np.uint8))
    spec = {"path": str(folder), "frames": 2, "start": 1}
    full = read_video(spec)
    crop = read_video(spec, [64, 96], torch.Generator().manual_seed(19))
    assert full.shape == (2, 3, 80, 120)
    assert crop.shape == (2, 3, 64, 96)
    torch.testing.assert_close(crop[0], torch.full_like(crop[0], 30 / 255), rtol=0, atol=0)
    with pytest.raises(ValueError, match="leakage"):
        check_splits(
            {
                "train": [dict(spec, content_id="same-scene")],
                "test": [dict(spec, path="elsewhere", content_id="same-scene")],
            }
        )


@pytest.mark.parametrize("height,width", [(256, 256), (512, 512), (540, 960), (720, 1280), (1080, 1920)])
def test_resolution_padding_preserves_native_pixels_and_latent_shape(height, width):
    from cp_adaptok.backbones.causal_state import FrozenFrameContext
    from cp_adaptok.backbones.intra_adapter import prepare_rgb, restore_rgb

    rgb = torch.rand(1, 3, height, width)
    padded = prepare_rgb(rgb)
    assert padded.shape[-2:] == ((height + 63) // 64 * 64, (width + 63) // 64 * 64)
    torch.testing.assert_close(restore_rgb(padded, height, width), rgb, atol=3e-7, rtol=1e-5)
    context = FrozenFrameContext("test", 0, "I", height, width, 32, -1, "00" * 16)
    assert context.latent_shape == ((height + 63) // 64 * 4, (width + 63) // 64 * 4)


def test_continuous_query_to_decoder_path_can_overfit(codec):
    model = copy.deepcopy(codec).train()
    model.vq.set_training_policy("continuous")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.005)
    y = torch.randn(1, 256, 4, 4)
    target = torch.randn_like(y)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        features = model.tokenizer(y, None, "I")
        embeddings = model.vq(features).embeddings
        reconstruction = model.decoder(embeddings, (4, 4), "I")
        loss = (reconstruction - target).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.9
    assert model.tokenizer.queries.grad.norm() > 0
    assert model.vq.project_in.weight.grad.norm() > 0


def test_optional_profiling_preserves_packets_and_strict_reference_state(codec):
    from cp_adaptok.profiling import PipelineProfiler
    from cp_adaptok.streaming.sender import CodecEncoder

    model = copy.deepcopy(codec).eval()
    frame = torch.rand(1, 3, 32, 48)
    expected = CodecEncoder(model).encode_frame(frame)
    model.pipeline_profiler = PipelineProfiler(model.device)
    sender = CodecEncoder(model)
    actual = sender.encode_frame(frame)
    assert actual.packets == expected.packets
    assert actual.token_ids == expected.token_ids
    assert actual.header_bytes == expected.header_bytes
    assert all(row["wall_ms"] >= 0 and row["cuda_ms"] is None for row in model.pipeline_profiler.records)
    assert any(row["name"] == "encoder.local_base_decode" for row in model.pipeline_profiler.records)


def test_initialization_collects_real_pre_vq_features_and_restores_training_policy(codec, tmp_path):
    import json

    import cv2
    import numpy as np

    from cp_adaptok.training_initialization import initialize_from_training_data

    model = copy.deepcopy(codec).train()
    source = tmp_path / "source"
    source.mkdir()
    generator = np.random.default_rng(19)
    for i in range(4):
        cv2.imwrite(str(source / f"{i:05d}.png"), generator.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    config = {
        "seed": 19,
        "data": {"train": [{"path": str(source), "frames": 4}]},
        "training": {"crop": 32, "vq": {"initialization": {"clips": 1, "max_samples": 128, "iterations": 2}}},
    }
    original = model.vq.codebook.weight.detach().clone()
    initialize_from_training_data(model, config, tmp_path)
    assert model.vq.strategy == "deterministic_nearest"
    assert not torch.equal(original, model.vq.codebook.weight)
    report = json.loads((tmp_path / "codebook_initialization.json").read_text())
    assert report["samples"] == 128
    assert report["distinct_samples"] >= 32
    assert report["source_split"] == "train"


def test_content_readout_cannot_transmit_a_query_identity_without_spatial_values():
    from cp_adaptok.tokenizer.lightweight_attention import CrossAttentionBlock

    block = CrossAttentionBlock(24, 4)
    with torch.no_grad():
        block.kv.weight.zero_()
        block.kv.bias.zero_()
        block.out.bias.zero_()
        for parameter in block.ff.parameters():
            parameter.zero_()
    queries, spatial = torch.randn(1, 32, 24), torch.randn(1, 8, 24)
    residual = block(queries, spatial)
    content = block(queries, spatial, include_query_residual=False)
    torch.testing.assert_close(residual, queries, rtol=0, atol=0)
    assert content.count_nonzero() == 0


def test_default_model_identity_remains_compatible_with_existing_packets(codec):
    import hashlib
    import json
    from dataclasses import asdict

    config = asdict(codec.config)
    config.pop("query_content_only")
    config.pop("token_aggregation")  # Default additions must not enter historical identity bytes.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, tensor in sorted(codec.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert codec.model_id == digest.hexdigest()[:32]


def test_online_budget_selection_cannot_read_actual_distortion_labels():
    from cp_adaptok.evaluation import allocation_comparison

    rows = [
        {
            "sequence": "clip",
            "frame_id": t,
            "fps": 30,
            "encode_timing_ms": {"encode_total_ms": 1},
            "prefixes": [
                {"tokens": 128, "total_bytes": 200, "mse": 0.1, "predicted_mse": 0.1},
                {"tokens": 2048, "total_bytes": 2000, "mse": 0.01, "predicted_mse": 0.02},
            ],
        }
        for t in range(2)
    ]
    changed = copy.deepcopy(rows)
    for frame in changed:
        for p in frame["prefixes"]:
            p["mse"] = 1 - p["mse"]
    first = allocation_comparison(rows, {}, 128)
    second = allocation_comparison(changed, {}, 128)
    assert [{k: v["tokens"] for k, v in r["online"].items()} for r in first] == [
        {k: v["tokens"] for k, v in r["online"].items()} for r in second
    ]


def test_checkpoint_restores_numpy_rng_alongside_torch_and_sampling_rng(codec, tmp_path):
    import numpy as np

    from cp_adaptok.runtime import load_checkpoint, save_checkpoint
    from cp_adaptok.training import restore_training

    model = copy.deepcopy(codec)
    optimizer = torch.optim.AdamW(model.vq.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(19)
    np.random.seed(19)
    save_checkpoint(tmp_path / "rng.pt", model, optimizer, generator, 1, {"stage": 1}, "test")
    expected = np.random.random(5)
    restore_training(model, optimizer, generator, load_checkpoint(tmp_path / "rng.pt"), 1)
    assert np.array_equal(np.random.random(5), expected)
