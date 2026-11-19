import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.models.codec import AIPVTokCodec
from aip_vtok.runtime import file_hash, resolve, save_checkpoint
from aip_vtok.tokenizer.fsq import FSQ


class SmallBackbone(nn.Module):
    def begin_frame(self, state, frame):
        intra = state.is_intra(frame, 32)
        return FrameContext(frame, intra, state.feature, None if intra else torch.ones(1))

    def aligned_latent(self, pixels, context):
        return pixels

    def decode_aligned(self, latent, context):
        self.decoded_context = context
        channels = 192 if context.intra else 256
        feature = F.interpolate(latent.mean(1, keepdim=True), scale_factor=2).expand(-1, channels, -1, -1)
        return latent, feature


@pytest.mark.parametrize("frame", [0, 1])
def test_reference_supervision_uses_the_exact_inference_prefix_with_tokenizer_gradients(frame):
    from aip_vtok.training.losses import reference_loss
    from aip_vtok.training.prefix import commit_training_reference, training_reference

    torch.manual_seed(71)
    model = AIPVTokCodec(SmallBackbone(), LatentToRef(8), reference_i=14, reference_p=8)
    state = ReferenceState() if frame == 0 else ReferenceState(frame_id=0)
    native = torch.randn(1, 128, 8, 16, requires_grad=True)
    encoded = model.encode_frame(native, state, frame)
    encoded.tokens.codes.retain_grad()
    reference = training_reference(model, encoded)
    budget = model.reference_i if frame == 0 else model.reference_p
    inference = model.prefix_session(encoded, state, budget)
    expected = inference.commit_reference_once()
    assert reference.latent.requires_grad and reference.budget == budget
    assert torch.equal(reference.latent.detach(), expected.latent)
    assert torch.equal(reference.feature.detach(), expected.feature)
    loss = reference_loss(model.backbone, encoded, reference)
    assert model.backbone.decoded_context is encoded.context
    loss.backward()
    assert model.tokenizer.anchor[0][0].weight.grad.norm() > 0
    assert model.tokenizer.lifts[0].weight.grad.norm() > 0
    assert encoded.tokens.codes.grad[:, :budget].abs().sum() > 0
    assert torch.count_nonzero(encoded.tokens.codes.grad[:, budget:]) == 0
    assert all(not p.requires_grad and p.grad is None for p in model.projector.parameters())
    assert native.grad is None
    committed = commit_training_reference(
        model, encoded.tokens, encoded.prediction, encoded.context, reference=reference
    )
    for actual, tensor in ((committed.latent, reference.latent), (committed.feature, reference.feature)):
        assert torch.equal(actual, tensor.detach())
        assert actual.grad_fn is None and not actual.requires_grad and actual.data_ptr() != tensor.data_ptr()


def test_fsq_distinguishes_boundary_codes_from_shifted_tanh_saturation():
    fsq = FSQ()
    values = torch.tensor([1.6, 2.1, 4.0]).view(1, 3, 1).expand(-1, -1, 6)
    codes, ids, saturated = fsq(values, measure_saturation=True)
    assert torch.equal(codes, fsq.ids_to_codes(ids))
    assert torch.all(fsq.ids_to_digits(ids)[:, (0, 2)] == fsq.levels - 1)
    assert saturated[0].tolist() == [[False] * 6, [True] * 3 + [False] * 3, [True] * 6]


def test_two_sided_chroma_gate_rejects_false_color_and_mean_shift():
    from aip_vtok.evaluation.fix_metrics import chroma_passes, fix_chroma

    baseline = {"chroma_var_retention_pct": 100, "chroma_mse": 0.001, "chroma_mean_bias": 0.01}
    assert chroma_passes(baseline)
    for key, value in (
        ("chroma_var_retention_pct", 4838),
        ("chroma_var_retention_pct", 74.9),
        ("chroma_mse", 0.0026),
        ("chroma_mean_bias", 0.0151),
    ):
        assert not chroma_passes({**baseline, key: value})
    for value in (75, 130):
        assert chroma_passes({**baseline, "chroma_var_retention_pct": value})
    rgb = torch.full((1, 3, 2, 2), 0.5)
    variation = torch.tensor([[-0.001, 0.001], [-0.001, 0.001]])
    distorted = rgb.clone()
    rgb[:, 2] += variation
    distorted[:, 2] += variation * 48.38**0.5
    measured = fix_chroma(rgb, distorted)
    assert measured["chroma_var_retention_pct"] == pytest.approx(4838, rel=0.002)
    assert measured["chroma_mse"] < 0.0025 and measured["chroma_mean_bias"] < 0.015
    assert not chroma_passes(measured)


def test_raw_and_fused_anchor_each_stop_gradient_at_native_target():
    from aip_vtok.training.losses import latent_losses

    native = torch.ones(4, requires_grad=True)
    raw = torch.zeros_like(native, requires_grad=True)
    fused = torch.full_like(native, 0.5, requires_grad=True)
    losses = latent_losses(raw, fused, native)
    assert losses["latent_raw"] == 1 and losses["latent_fused"] == 0.5
    sum(losses.values()).backward()
    assert native.grad is None
    assert torch.all(raw.grad == -0.25) and torch.all(fused.grad == -0.25)


def test_pooled_fsq_entropy_is_computed_from_counts_and_boundary_codes_are_descriptive():
    from aip_vtok.evaluation.fix_gate import healthy_fsq
    from aip_vtok.evaluation.fix_metrics import fsq_stats, pooled_fsq

    rows = []
    for digit, identifier in ((0, 0), (-1, 63999)):
        dims = [[0] * n for n in (8, 8, 8, 5, 5, 5)]
        for hist in dims:
            hist[digit] = 2
        stats = {"id_counts": [(identifier, 2)], "dim_counts": dims, "saturated_elements": 0}
        stats.update(fsq_stats([2], dims, 0))
        rows.append({"passes": {str(p): stats for p in range(3)}})
    pooled = pooled_fsq(rows)
    for item in pooled.values():
        assert item["pooled"]["dim_entropy_bits"] == [1] * 6
        assert item["pooled"]["effective_ids"] == 2
        assert item["pooled"]["boundary_code_ratio"] == 1
        assert item["per_frame"]["dim_entropy_bits"]["p05"] == [0] * 6
        assert item["per_frame"]["effective_ids"]["min"] == 1
    limits = {"maximum_tanh_saturation_ratio": 0.1, "minimum_pooled_dim_entropy_bits": 0.5}
    assert healthy_fsq(pooled, limits)
    pooled["2"]["pooled"]["tanh_saturation_ratio"] = 0.1001
    assert not healthy_fsq(pooled, limits)


def test_joint_projector_checkpoints_round_trip_and_leave_gate0_weights_untouched(tmp_path):
    from aip_vtok.training.checkpoint import checkpoint_artifact, load_fix_weights
    from aip_vtok.training.components import enable_projector
    from aip_vtok.training.fix import directories, read_json
    from aip_vtok.training.trainer import RNGStreams

    source = resolve("checkpoints/aip_vtok/projector_w128/latest.pt")
    if not source.is_file():
        pytest.skip("sealed Gate 0 weights required for the isolation audit")
    sealed = read_json(resolve("results/aip_vtok/checkpoints_manifest.json"))
    base_hash = file_hash(source)
    assert base_hash == sealed["projector_w128/latest.pt"]["sha256"]
    assert directories("b1", 1)[1].parent == resolve("checkpoints/aip_vtok/gate1_fix/fix_b")
    original = torch.load(source, map_location="cpu", weights_only=False)["projector"]
    model = AIPVTokCodec(SmallBackbone(), LatentToRef())
    model.projector.load_state_dict(original)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    enable_projector(model, optimizer, params)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4)
    project = model.projector(torch.ones(1, 128, 4, 4), True)
    project.square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(original[name], param) for name, param in model.projector.state_dict().items())
    config = {"model_version": model.model_version, "joint_projector": True}
    artifact = checkpoint_artifact(1, model, optimizer, RNGStreams(1), [], config, {})
    isolated = tmp_path / "fix_b" / "latest.pt"
    save_checkpoint(isolated, artifact)
    restored = AIPVTokCodec(SmallBackbone(), LatentToRef())
    loaded = torch.load(isolated, map_location="cpu", weights_only=False)
    load_fix_weights(restored, loaded)
    assert all(
        torch.equal(a, b) for a, b in zip(model.projector.parameters(), restored.projector.parameters())
    )
    assert file_hash(source) == base_hash
    with pytest.raises(ValueError, match="version mismatch"):
        load_fix_weights(restored, {**loaded, "model_version": "gate1-508a97e"})
    with pytest.raises(ValueError, match="policy"):
        load_fix_weights(restored, {k: v for k, v in loaded.items() if k != "projector"})


def test_continuous_oracle_recovers_a_signal_in_the_frozen_lift_subspace():
    from aip_vtok.evaluation.tokenizer_oracle import continuous_codes, unpack_patches
    from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer

    tokens = ResidualTokenizer()
    shape = (8, 12)
    for pass_id, size in enumerate((4, 2, 1)):
        pos = torch.arange(shape[0] // size * (shape[1] // size))
        lift = tokens.lifts[pass_id]
        codes = torch.randn(1, pos.numel(), 6)
        latent = tokens.synthesize_pass(codes, pos, pass_id, shape)
        inverse = torch.linalg.pinv(lift.weight.double()).float()
        recovered = continuous_codes(lift, unpack_patches(latent, size), inverse)
        torch.testing.assert_close(recovered, codes, atol=1e-5, rtol=1e-5)


def test_gate1_fix_generated_section_preserves_a_sealed_prefix(tmp_path):
    from aip_vtok.evaluation.report_fix import BEGIN, replace_section

    document = tmp_path / "report.md"
    old = "# Historical\n\nSealed evidence.\n"
    document.write_text(old)
    replace_section(document, ["First pass"])
    replace_section(document, ["Second pass"])
    assert document.read_text().startswith(old + "\n" + BEGIN)
    assert document.read_text().count(BEGIN) == 1
    assert "Second pass" in document.read_text()
    assert "First pass" not in document.read_text()


def test_fix_a_failure_prevents_both_projector_branches_and_fix_c(tmp_path, monkeypatch):
    from aip_vtok.runtime import write_json
    from aip_vtok.training import fix

    monkeypatch.setattr(fix, "RESULTS", tmp_path)
    write_json(tmp_path / "fix_a/decision.json", {"passed": False})
    for phase in ("b0", "b1", "fix_c"):
        with pytest.raises(RuntimeError, match="requires a passing"):
            fix.require_release(phase)
