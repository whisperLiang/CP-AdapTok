"""One owner for the paired prefix sampling law."""

import torch


def sample_budgets(rng):
    anchor = 256 if torch.rand((), generator=rng.budget_rng) < 0.2 else 510
    middle = int(torch.randint(768, 2049, (), generator=rng.budget_rng))
    detail = int(torch.randint(3072, 8193, (), generator=rng.budget_rng))
    return [anchor, middle, detail]
