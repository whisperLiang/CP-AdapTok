import torch

from experiments.fixed2048.calibrate import collect_train_features
from experiments.fixed2048.tests.test_codec import make_model


def test_calibration_receives_only_declared_training_sources(monkeypatch):
    import experiments.fixed2048.calibrate as calibration

    sources = [{"content_id": "train:a"}, {"content_id": "train:b"}]
    monkeypatch.setattr(calibration, "training_sources", lambda config: (sources, {"train": "frozen"}))
    monkeypatch.setattr(calibration, "read_clip", lambda *a, **k: torch.rand(4, 3, 64, 64))
    samples, record = collect_train_features(make_model().eval(), {}, videos=2, frames=4)
    assert samples.shape == (2 * 4 * 2048, 16)
    assert record["sample_count"] == len(samples)
    assert all(r["split"] == "train" for r in record["samples"])
