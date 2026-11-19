import pytest
import torch
from torch.nn import functional as F

from aip_vtok.training.losses import native_global_view


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_global_view_matches_bilinear_sampling_and_exact_cpu_gradient(dtype):
    rgb = torch.rand(1, 3, 1080, 1920, dtype=dtype, requires_grad=True)
    reference = rgb.detach().clone().requires_grad_(True)
    actual = native_global_view(rgb)
    expected = F.interpolate(reference, size=(270, 480), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.equal(rgb.grad, reference.grad)
