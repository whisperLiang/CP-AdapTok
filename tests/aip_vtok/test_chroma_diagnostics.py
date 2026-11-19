import pytest
import torch

from aip_vtok.training.losses import chroma_diagnostics


def test_chroma_diagnostics_separate_variance_from_global_mean_color():
    rgb = torch.ones(1, 3, 32, 32) * 0.5
    rgb[:, 0, :, :16] += 0.2
    rgb[:, 0, :, 16:] -= 0.2
    reduced = 0.5 + (rgb - 0.5) * 0.5
    values = chroma_diagnostics(rgb, reduced)
    assert values["chroma_retention"] == pytest.approx(0.25, abs=1e-6)
    assert values["chroma_mse"] > 0
    assert values["chroma_mean_abs_bias"] < 1e-6
