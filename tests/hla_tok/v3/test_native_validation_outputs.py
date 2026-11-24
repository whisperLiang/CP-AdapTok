import torch

from hla_tok import Geometry, HLATok
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3 import quality


def test_quantized_validation_reports_bytes_only_for_the_independent_receiver_path(tmp_path, monkeypatch):
    geometry = Geometry(128, 128)
    generator = torch.Generator().manual_seed(7)
    raw = torch.rand((3, 128, 128), generator=generator)

    class Reader:
        counts = {"train": 0, "validation": 0, "test": 0}

        def read(self, split, video, index):
            assert split == "validation"
            self.counts[split] += 1
            return raw

    def read_quality(source, reconstruction):
        error = reconstruction - source
        assert reconstruction.shape == source.shape
        mse = float(error.square().mean())
        score = quality.psnr(mse)
        return {"mse": mse, "psnr": score, "lpips": mse, "lpips_vgg": mse, "dists": mse,
                "ms-ssim": 1 - mse, "boundary_mse": mse, "interior_mse": mse, "boundary_minus_interior_mse": 0.0}

    monkeypatch.setattr(quality, "NATIVE", geometry)
    monkeypatch.setattr(quality, "transfer", lambda rgb: rgb)
    model = HLATok("R2")
    report = quality.evaluate(model, Reader(), {"a": 11, "b": 11, "c": 11}, read_quality, tmp_path,
                              budgets=(32,), quantized=True, subset=True, bypass=True)
    fixed, bypass = report
    assert (fixed["mode"], bypass["mode"]) == ("fixed13", "same_checkpoint_bypass")
    assert fixed["total_bytes"] == 52 * 9
    assert fixed["rate_basis"] == "actual_fixed13_payload"
    assert fixed["quantization_mse"] > 0
    assert "actual_bpp" not in bypass and "total_bytes" not in bypass
    assert fixed["frames"] == bypass["frames"] == 9
    assert not fixed["matched_174"] and not bypass["matched_174"]
    assert len(csv_at(tmp_path / "per_frame.csv")) == 18
    assert json_at(tmp_path / "pixel_reads.json") == {"train": 0, "test": 0, "validation": 9}
