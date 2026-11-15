import torch


def prefix_mask(lengths, n_max):
    if lengths.ndim != 1 or (lengths < 1).any() or (lengths > n_max).any():
        raise ValueError("prefix lengths must be in [1,N_max]")
    return torch.arange(n_max, device=lengths.device)[None] < lengths[:, None]


def sample_budgets(candidates, generator, count=None):
    """Log-spaced candidates with twice the endpoint probability, without replacement."""
    if len(candidates) < 2 or sorted(set(candidates)) != list(candidates):
        raise ValueError("need at least two increasing unique budgets")
    count = (
        int(torch.randint(2, min(4, len(candidates)) + 1, (), generator=generator))
        if count is None
        else count
    )
    weights = torch.ones(len(candidates))
    weights[0] = weights[-1] = 2
    indices = torch.multinomial(weights, count, replacement=False, generator=generator).tolist()
    return sorted(candidates[i] for i in indices)


def sample_nested_budgets(candidates, generator, sample_intermediate=1):
    """One EncodeOnce sequence, with reference and maximum supervised every update."""
    if len(candidates) < 2 or sorted(set(candidates)) != list(candidates):
        raise ValueError("need at least two increasing unique budgets")
    if type(sample_intermediate) is not int or sample_intermediate < 1:
        raise ValueError("at least one intermediate budget must be requested")
    middle = candidates[1:-1]
    indices = torch.randperm(len(middle), generator=generator)[:sample_intermediate].tolist()
    return sorted([candidates[0], candidates[-1], *(middle[i] for i in indices)])
