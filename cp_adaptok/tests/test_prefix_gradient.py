import pytest
import torch

from cp_adaptok.tokenizer.prefix_masking import sample_nested_budgets
from cp_adaptok.training import prefix_distortion, progressive_penalty, validate_prefix_training


def test_nested_sampling_always_supervises_base_max_and_an_intermediate():
    generator = torch.Generator().manual_seed(7)
    samples = [sample_nested_budgets([8, 16, 24, 32], generator, 1) for _ in range(30)]
    assert all(len(row) == 3 and row[0] == 8 and row[-1] == 32 for row in samples)
    assert {row[1] for row in samples} == {16, 24}
    with pytest.raises(ValueError):
        sample_nested_budgets([8, 16, 32], generator, 0)


def test_one_sided_progress_penalty_only_pushes_high_branch():
    low, high = torch.tensor(0.1, requires_grad=True), torch.tensor(0.2, requires_grad=True)
    penalty = progressive_penalty([low, high], {"mode": "progressive_nested", "margin": 0})
    a, b = torch.autograd.grad(penalty, (low, high), allow_unused=True)
    assert a is None or a == 0
    assert b == 1
    absolute = prefix_distortion([low, high], [8, 32], {})
    a, b = torch.autograd.grad(absolute, (low, high))
    assert a > 0 and b > 0


def test_explicit_weights_are_normalized_by_active_budget_count():
    low, high = torch.tensor(2.0, requires_grad=True), torch.tensor(5.0, requires_grad=True)
    settings = {"weights": {8: 0.5, 16: 99.0, 32: 1.5}}
    loss = prefix_distortion([low, high], [8, 32], settings)
    assert loss == (low * 0.5 + high * 1.5) / 2
    a, b = torch.autograd.grad(loss, (low, high))
    assert a == 0.25 and b == 0.75
    with pytest.raises(ValueError, match="positive"):
        prefix_distortion([low, high], [8, 32], {"weights": {8: 0, 32: 1}})


def test_scheduled_weights_use_only_reached_training_stage():
    values = [torch.tensor(2.0), torch.tensor(5.0)]
    settings = {"weights": {8: 1, 32: 1}, "weight_schedule": [{"step": 10, "weights": {8: 0.5, 32: 1.5}}]}
    assert prefix_distortion(values, [8, 32], settings, 0) == 3.5
    assert prefix_distortion(values, [8, 32], settings, 10) == 4.25


def test_invalid_prefix_training_is_rejected_before_updating_parameters(codec):
    import copy

    config = {"training": {"budgets": [8, 16, 32]},
              "prefix_training": {"mode": "progressive_nested", "budgets": [8, 16, 32],
                                  "weights": {8: 1, 16: 1, 32: 1}}}
    validate_prefix_training(codec, config)
    for change in ({"mode": "typo"}, {"budgets": [8, 32]}, {"always_include_base": False},
                   {"margin": -0.1}, {"weights": {8: 1, 32: 1}}):
        invalid = copy.deepcopy(config)
        invalid["prefix_training"].update(change)
        with pytest.raises(ValueError):
            validate_prefix_training(codec, invalid)
