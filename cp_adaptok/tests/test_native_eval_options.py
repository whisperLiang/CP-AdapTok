import json

import torch

from cp_adaptok.backbones import native_baseline
from cp_adaptok.cli import eval as cli
from cp_adaptok.runtime import read_config


class FakeNative:
    def __init__(self, *args):
        self.reset()

    def reset(self):
        self.frame_id = 0

    def round_trip(self, rgb):
        self.frame_id += 1
        return rgb, {"total_bytes": 20, "payload_bytes": 10}, b"x" * 20


class FakeMetrics:
    def __init__(self, device, names=None):
        self.names = names or ["psnr", "lpips"]
        self.errors = {}

    def __call__(self, rgb, rec):
        return {name: 1.0 for name in self.names}


def test_native_eval_honors_output_split_frame_limit_and_metrics(monkeypatch, tmp_path):
    config = read_config("configs/pretrained_s1.yaml")
    config["paths"]["output"] = str(tmp_path / "default")
    monkeypatch.setattr(cli, "read_config", lambda _: config)
    monkeypatch.setattr(cli, "preflight", lambda *args: None)
    monkeypatch.setattr(cli, "read_video", lambda spec: torch.zeros(spec["frames"], 3, 16, 16))
    monkeypatch.setattr(cli, "MetricSuite", FakeMetrics)
    monkeypatch.setattr(native_baseline, "NativeBaseline", FakeNative)
    requested = tmp_path / "requested"
    assert (
        cli.main(
            [
                "--config",
                "unused.yaml",
                "--baseline",
                "native",
                "--device",
                "cpu",
                "--output",
                str(requested),
                "--split",
                "validation",
                "--max-frames",
                "3",
                "--metrics",
                "psnr",
            ]
        )
        == 0
    )
    report = json.loads((requested / "eval/native_baseline.json").read_text())
    assert len(report["records"]) == 3 and report["split"] == "validation"
    assert all(row["source"] == config["data"]["validation"][0]["path"] for row in report["records"])
    assert all("psnr" in row and "lpips" not in row for row in report["records"])
    assert not (tmp_path / "default/native_baseline.json").exists()
