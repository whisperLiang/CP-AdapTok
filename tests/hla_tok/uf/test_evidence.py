import pytest

from scripts.hla_tok.uf.dynamic import dynamic_summary
from scripts.hla_tok.uf.evidence import interpolate, worsening


def test_rate_comparison_does_not_extrapolate_or_pad_rates():
    curve = [(1.0, 20.0), (4.0, 28.0)]
    assert interpolate(curve, 2.0, log_x=True) == pytest.approx(24.0)
    assert interpolate(curve, 8.0, log_x=True) is None
    assert interpolate(curve, 0.0, log_x=True) is None
    assert interpolate(curve, 4.0, log_x=True) == 28.0


def test_frame_violations_are_not_hidden_by_an_improved_average():
    assert worsening([30.0, 20.0], [29.0, 22.0], higher=True) == {"violations": 1, "fraction": 0.5, "max_drop": 1.0}
    assert worsening([0.1, 0.2], [0.3, 0.0], higher=False)["violations"] == 1
    with pytest.raises(ValueError):
        worsening([0.1], [], higher=False)


def test_dynamic_status_rejects_base_and_full_packet_overages_even_with_exact_reference():
    row = {"video": "held-out-sequence", "type": "I", "budget_mbps": 5, "base_bytes": 9212,
           "base_cap_pass": True, "state_exact": True, "bytes": 20833}
    counts = {"train": 0, "validation": 1, "test": 0}
    assert dynamic_summary([row], counts)["status"] == "PASS"
    base_overflow = dynamic_summary([row | {"base_bytes": 9213, "base_cap_pass": False}], counts)
    assert base_overflow["status"] == "FAIL"
    assert base_overflow["base_exceeded"] == 1
    packet_overflow = dynamic_summary([row | {"bytes": 20834}], counts)
    assert packet_overflow["status"] == "FAIL"
    assert packet_overflow["packet_exceeded"] == 1
