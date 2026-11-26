from pht_tok.streaming import Observation, RateTrace, Transport, goodput_at, select_prefix


def test_prefix_fits_exactly_with_codec_and_transport_cost():
    boundaries = (100, 230, 610, 1200)
    assert select_prefix(boundaries, 16000, 1, 0.1, 0.1, 0.05, minimum=2) == 4
    assert select_prefix(boundaries, 16000, 0.41, 0.1, 0.1, 0.05, minimum=2) == 2
    assert select_prefix(boundaries, 8000, 0.4, 0.1, 0.1, 0.05, minimum=2) == 0


def test_feedback_never_uses_future_observations_and_tardy_packets_are_charged():
    feedback = (Observation(0.4, 2_000_000), Observation(0.6, 100_000_000))
    assert goodput_at(feedback, 0.5, 1_000_000) == goodput_at(feedback[:1], 0.5, 1_000_000)
    trace = RateTrace(((0.0, 1.0), (0.25, 0.1), (0.6, 1.0)))
    link = Transport(trace, seed=123, rtt_seconds=0.02)
    result = link.send(0.24, (100, 4000))
    assert result.bytes_sent > 4000
    assert result.received_by(0.45) < result.bytes_received
    assert result.arrivals[0] < result.arrivals[1]
    assert result.stage_by(0.45, 0) == 1
    assert result.observation.at > result.arrivals[-1]


def test_trace_rejects_missing_or_nonphysical_measurements():
    import pytest

    illegal = ((), ((1, 1),), ((0, 0),), ((0, float("nan")),), ((0, 1), (0, 2)))
    for measurement in illegal:
        with pytest.raises(ValueError):
            RateTrace(measurement)
