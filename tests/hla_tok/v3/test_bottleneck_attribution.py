import csv
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hla_tok import Geometry
from hla_tok.progressive.geometry import FEATURE_SIDE
from scripts.hla_tok.v3 import diagnostics


def test_latent_alignment_weights_partial_valid_cells_and_internal_region_seams():
    geometry = Geometry(1080, 1920)
    height = geometry.rows * FEATURE_SIDE
    width = geometry.columns * FEATURE_SIDE
    reference = torch.arange(width, dtype=torch.float32).expand(1, 1, height, width).clone()
    recovered = reference.clone()
    recovered[..., 67, :] += 2
    recovered[..., 68:, :] += 4

    rows, variance = diagnostics.latent_alignment(reference, recovered, geometry)
    by_scope = {row["region"]: row for row in rows}
    assert by_scope["valid"]["mse"] == pytest.approx(2 / 67.5)
    assert by_scope["padding"]["mse"] == pytest.approx(66 / 4.5)
    assert by_scope["valid"]["l1"] == pytest.approx(1 / 67.5)
    assert by_scope["valid"]["cosine_similarity"] < 1
    assert by_scope["valid"]["channel_normalized_rmse"] < by_scope["padding"]["channel_normalized_rmse"]
    assert by_scope["whole"]["latent_error_power_low"] > 0
    assert by_scope["internal_seam"]["mse"] != by_scope["interior"]["mse"]
    assert {(row["region"], row["channel"]) for row in variance} == {("valid", 0), ("padding", 0)}
    assert all(row["reference_variance"] == pytest.approx((width**2 - 1) / 12) for row in variance)


def test_diagnosis_uses_one_b0_encode_per_frame_and_records_intervention_counts(tmp_path, monkeypatch):
    geometry = Geometry(128, 256)
    raw = torch.ones(3, geometry.height, geometry.width)
    source_features = torch.ones(1, 2, FEATURE_SIDE, FEATURE_SIDE * geometry.columns)

    class Encoder(nn.Module):
        def forward(self, source, received_geometry):
            assert received_geometry == geometry
            return None, source_features * 2

    class Dense(nn.Module):
        def coarse(self, source):
            return source_features, geometry

        def rgb_decoder(self, feature, received_geometry):
            return raw[None] * feature.mean()

    class Student(Dense):
        encoder = Encoder()

        def encode_training(self, source, budget, *, quantize):
            assert not quantize
            self.encoder(source, geometry)
            assert budget == 244
            return SimpleNamespace(values=torch.ones(1, budget, 2), geometry=geometry)

        def _decode_features(self, values, received_geometry):
            return source_features * 3

    class Reader:
        counts = dict(train=0, validation=0, test=0)

        def read(self, split, video, index):
            assert split == "validation"
            self.counts[split] += 1
            return raw

    intervention_calls = []

    def intervention(model, values, received_geometry, case):
        intervention_calls.append((model, case))
        count = 244 - (32 if case.startswith("delete_") else 0)
        return raw[None] * 4, count

    def quality(source, recovered):
        result = recovered.mean().item()
        return {key: result for key in ("mse", "psnr", "lpips", "lpips_vgg", "dists", "ms-ssim",
                                         "boundary_mse", "interior_mse", "boundary_minus_interior_mse")}

    monkeypatch.setattr(diagnostics, "transfer", lambda raw: raw)
    monkeypatch.setattr(diagnostics, "reconstruct_intervention", intervention)
    student = Student()
    with torch.inference_mode():
        rows = diagnostics.diagnose(Dense(), student, Reader(), {"v1": 6, "v2": 6, "v3": 6},
                                    quality, 244, tmp_path, subset=True)
    by_mode = {row["mode"]: row for row in rows}
    assert [by_mode[key]["mse"] for key in ("phase_a_dense", "b0_dense_through_b0_rgb", "b0_local")] == [1, 2, 3]
    assert all(row["frames"] == 9 and not row["matched_174"] for row in rows)
    assert by_mode["repeat_rank0"]["intervention"] == "token_content"
    assert by_mode["perturb_rank_assignment"]["intervention"] == "rank_assignment"
    assert by_mode["delete_middle16"]["intervention"] == "token_count"
    assert by_mode["delete_late16"]["received_tokens"] == 212
    assert len(intervention_calls) == 9 * 7
    assert all(model is student for model, _ in intervention_calls)
    assert not student.encoder._forward_hooks

    with (tmp_path / "latent_alignment_metrics.csv").open() as stream:
        alignment = list(csv.DictReader(stream))
    assert {row["comparison"] for row in alignment} == {"phase_a_vs_b0_dense", "b0_dense_vs_local", "phase_a_vs_b0_local"}
    with (tmp_path / "token_representation.csv").open() as stream:
        representation = list(csv.DictReader(stream))
    assert len(representation) == 9
    assert float(representation[0]["common_energy_fraction"]) == pytest.approx(1)
    assert (tmp_path / "pixel_reads.json").read_text().count('"validation": 9') == 1


def test_diagnose_can_select_a_completed_checkpoint_before_b0_admission(monkeypatch):
    import sys

    from scripts.hla_tok.v3.run import _arguments

    monkeypatch.setattr(sys, "argv", ["run", "diagnose", "--seed", "1", "--step", "12000", "--full"])
    args = _arguments()
    assert (args.seed, args.step, args.full) == (1, 12000, True)


def test_gradient_source_rejects_a_different_frozen_plan(tmp_path, monkeypatch):
    from scripts.hla_tok.v3 import migration_audit
    from scripts.hla_tok.v3.protocol import Trial, encode_record, extend_chain

    monkeypatch.setattr(migration_audit, "OUTPUT", tmp_path)
    source = tmp_path / "plans/continuous_seed1.jsonl"
    source.parent.mkdir()
    trial = Trial("continuous", "B0", 1, "D0")
    samples = [{"split": "train", "frame": 2}, {"split": "train", "frame": 3}]
    chain = extend_chain(extend_chain(bytes(32).hex(), samples[0]), samples[1])
    state = {"step": 2, "counters": {"sample_chain": chain}}
    source.write_bytes(b"\n".join(encode_record(sample) for sample in samples) + b"\n")
    assert migration_audit._gradient_sample(trial, state) == samples[1]
    samples[0]["frame"] = 9
    source.write_bytes(b"\n".join(encode_record(sample) for sample in samples) + b"\n")
    with pytest.raises(ValueError, match="checkpoint's Train sample chain"):
        migration_audit._gradient_sample(trial, state)
