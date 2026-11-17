import math

import torch

from experiments.original_adaptok_1080p.metrics import aggregate, boundary_mask, pixel_metrics


def test_psnr_aggregation_keeps_distinct_definitions():
    result = aggregate([{"mse": 0.01, "psnr": 20.0}, {"mse": 0.0001, "psnr": 40.0}])
    assert result["psnr_mean_db"] == 30
    assert result["psnr_from_global_mse_db"] == -10 * math.log10(0.00505)
    assert result["bpp"] is None and result["quality_only"]


def test_boundary_error_is_measured_on_the_native_image():
    mask = boundary_mask(12, 20, spacing=8, half_width=1)
    original = torch.zeros(1, 3, 12, 20)
    recon = mask.float()[None, None].expand_as(original)
    result = pixel_metrics(original, recon, mask)
    assert result["boundary_mse"] == 1
    assert result["interior_mse"] == 0
    assert result["boundary_minus_interior_mse"] == 1


def test_identity_has_infinite_psnr():
    x = torch.rand(1, 3, 12, 20)
    assert math.isinf(pixel_metrics(x, x, boundary_mask(12, 20, 8, 1))["psnr"])


def test_boundary_strips_have_declared_width_and_exclude_image_edges():
    mask = boundary_mask(300, 300, spacing=128, half_width=2)
    assert mask[10, 126:130].all()
    assert not mask[10, 125] and not mask[10, 130]
    assert mask[254:258, 10].all()
    assert not mask[0, 0] and not mask[299, 299]
