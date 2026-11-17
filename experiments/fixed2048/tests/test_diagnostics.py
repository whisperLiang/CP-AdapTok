import pytest
import torch

from experiments.fixed2048.diagnostics import aggregate, content_variants, token_statistics


def test_global_mse_psnr_and_video_mean_are_explicit():
    rows = [
        {"video": "a", "frame_type": "I", "mse": 0.01, "pixels": 10, "psnr": 20.0},
        {"video": "b", "frame_type": "P", "mse": 0.1, "pixels": 30, "psnr": 10.0},
    ]
    summary = aggregate(rows)
    assert summary["mean_frame_psnr"] == 15.0
    assert summary["global_mse"] == pytest.approx(0.0775)
    assert summary["global_mse_psnr"] == pytest.approx(-10 * __import__("math").log10(0.0775))
    assert summary["I"]["mean_frame_psnr"] == 20.0
    assert summary["P"]["mean_frame_psnr"] == 10.0
    assert len(summary["videos"]) == 2


def test_content_variants_change_content_preserve_shapes_and_normal():
    features = torch.randn(1, 2048, 192)
    other = torch.randn_like(features)
    variants = content_variants(features, other)
    assert set(variants) == {"normal", "constant", "shuffled", "other_frame"}
    assert torch.equal(variants["normal"], features)
    assert torch.equal(variants["other_frame"], other)
    assert torch.equal(variants["constant"][:, :1], variants["constant"][:, -1:])
    assert not torch.equal(variants["shuffled"], features)
    assert torch.equal(features.sort(dim=1).values, variants["shuffled"].sort(dim=1).values)
    stats = token_statistics(features)
    assert stats["continuous_variance"] > 0
    assert len(stats["group_variance"]) == 16
