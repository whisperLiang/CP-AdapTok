import pytest

from hla_tok import BUDGETS
from hla_tok.streaming.fixed13 import payload_ledger
from scripts.hla_tok.v3 import sealed
from scripts.hla_tok.v3.gates import (
    discrete_thresholds,
    progressive_decision,
    quality_pass,
    supports_extension,
)
from scripts.hla_tok.v3.protocol import PROTOCOL, read_config


def test_only_full_native_quality_can_pass_and_allow_a_bounded_extension():
    thresholds = {"global_psnr": 26, "lpips": 0.18, "dists": 0.08}
    better = {"global_psnr": 25.85, "lpips": 0.186, "dists": 0.081, "frames": 174, "matched_174": True}
    older = better | {"global_psnr": 25.2, "lpips": 0.22, "dists": 0.089}
    near = {"global_psnr": 0.5, "lpips": 0.03, "dists": 0.02}
    with pytest.raises(ValueError):
        quality_pass(better | {"frames": 9, "matched_174": False}, thresholds)
    assert supports_extension([older, better], thresholds, near)
    assert not supports_extension([older, better | {"lpips": 0.221}], thresholds, near)
    assert not supports_extension([older, better | {"dists": 0.101}], thresholds, near)


def test_progressive_gates_require_real_bytes_and_detect_top_budget_saturation():
    config = read_config(PROTOCOL)
    good = [{"budget": budget, "frames": 174, "matched_174": True, "mode": "fixed13",
             "global_psnr": 22.2 + rank * 0.76, "lpips": 0.235 - rank * 0.011, "dists": 0.11 - rank * 0.009}
            | payload_ledger(((budget * 13 + 7) // 8) * 174, budget * 174, 174, 1080, 1920)
            for rank, budget in enumerate(BUDGETS)]
    assert progressive_decision(good, config, [])["status"] == "PASS"
    plateau = good[:6] + [good[5] | {"budget": budget} for budget in BUDGETS[-2:]]
    assert not progressive_decision(plateau, config, [])["checks"]["not_saturated_in_top_three"]
    with pytest.raises(ValueError, match="real Fixed13"):
        discrete_thresholds(good[4] | {"mode": "continuous"}, config, [])
    with pytest.raises(ValueError, match="bracket"):
        discrete_thresholds(good[4] | {"actual_bpp": 0.5}, config, [])


def test_sealed_test_fails_before_evaluating_or_opening_any_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed, "FREEZE", tmp_path / "missing.json")
    with pytest.raises(PermissionError, match="remains sealed"):
        sealed.evaluate_sealed(read_config(PROTOCOL))
