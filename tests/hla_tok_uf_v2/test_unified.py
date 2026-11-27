import pytest
import torch

from baselines.hla_tok_uf_v2.latent import LATENT_UNIT, UnifiedRefinement, normalized_error


def test_refines_the_same_native_y_with_two_nested_steps():
    core = UnifiedRefinement()
    decoded = torch.full((1, 128, 2, 2), 6.0)
    native_step = torch.tensor(2.0)
    residual = torch.tensor([-0.74, -0.13, 0.24, 0.66]).reshape(1, 1, 2, 2)
    latent = decoded / native_step + residual
    error = normalized_error(latent, decoded, native_step)
    quantizer = core.quantizer(False)
    coarse, fine = quantizer.symbols(error)
    assert set(fine.flatten().tolist()) <= {-1, 0, 1}
    step = quantizer.step() * LATENT_UNIT
    estimate = decoded + (coarse + fine * 0.5) * step * native_step
    assert (latent * native_step - estimate).abs().max() <= (step * native_step / 4).max() + 1e-6
    assert torch.equal(decoded, torch.full_like(decoded, 6.0))


def test_native_scale_is_not_silently_saturated():
    decoded = torch.zeros((1, 256, 1, 1))
    core = UnifiedRefinement()
    with pytest.raises(ValueError, match="alphabet"):
        core.quantizer(True).symbols(normalized_error(decoded + 1000, decoded, torch.tensor(0.5)))
    with pytest.raises(ValueError, match="alphabet"):
        core.quantizer(True).symbols(normalized_error(decoded + 1, decoded, torch.tensor(0.0)))
