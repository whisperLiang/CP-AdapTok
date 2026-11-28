from types import SimpleNamespace

from baselines.adaptok import cli


def test_suite_covers_same_video_and_all_sealed_samples(tmp_path, monkeypatch):
    sources = [{"content_id": s} for s in ["cows", "goat", "parkour"]]
    config = {"start": 0, "frames": 64, "budgets": [512, 1024, 2048]}
    adapter = SimpleNamespace(input_size=128, frame_num=16, total_tokens=2048)
    calls = []

    def execute(adapter, evaluator, selected, output, stride, budgets, frames, start, config):
        calls.append((selected, stride, budgets, frames, start))
        return {"executed": True}

    monkeypatch.setattr(cli, "execute", execute)
    results = cli.run_suite(adapter, None, sources, tmp_path, config)
    assert len(results) == 4
    assert calls[:2] == [(sources[:1], 128, [2048], 16, 0), (sources[:1], 96, [2048], 16, 0)]
    assert calls[2:] == [(sources, 128, [512, 1024, 2048], 64, 0), (sources, 96, [512, 1024, 2048], 64, 0)]
