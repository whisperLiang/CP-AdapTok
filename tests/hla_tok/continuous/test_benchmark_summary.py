from scripts.hla_tok.continuous.benchmark import latency_summary


def test_latency_p95_and_p99_are_measured_quantiles():
    sample = list(range(1, 101))
    summary = latency_summary(sample)
    assert summary["mean_ms"] == 50.5
    assert summary["p50_ms"] == 50.5
    assert summary["p95_ms"] == 95.05
    assert summary["p99_ms"] == 99.01
    assert summary["fps_from_mean_ms"] == 1000 / 50.5
