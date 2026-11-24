import torch

from hla_tok.evaluation.representation_diagnostics import (
    centered_spectrum,
    distinct_cosine,
    representation_metrics,
)


def test_diagnostics_separate_common_energy_from_complementary_directions():
    opposite = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    shifted = (opposite + torch.tensor([0., 0., 100.]))[None]
    report = representation_metrics(shifted)
    assert report["raw_distinct_cosine"].item() > 0.999
    assert report["centered_distinct_cosine"].item() < 0
    torch.testing.assert_close(report["effective_rank"], torch.tensor([2.]))
    torch.testing.assert_close(report["stable_rank"], torch.tensor([2.]))
    torch.testing.assert_close(report["centered_variance"], torch.tensor([1 / 3]))
    assert report["common_energy_fraction"].item() > 0.999


def test_zero_and_near_zero_centered_matrix_has_no_inferred_rank():
    base = torch.ones(1, 5, 3)
    base[:, 1, 0] += 1e-7
    singular, effective, stable = centered_spectrum(base - base.mean(-2, keepdim=True))
    assert singular.max() < 1e-6
    assert effective.item() == stable.item() == 0
    assert distinct_cosine(torch.zeros(1, 3, 2)).item() == 0
