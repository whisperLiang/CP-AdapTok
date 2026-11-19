"""Integer-defined, budget-independent Halton pixel permutations."""

from functools import lru_cache

import torch


def radical_inverse(index, base):
    numerator, denominator = 0, 1
    while index:
        index, digit = divmod(index, base)
        numerator = numerator * base + digit
        denominator *= base
    return numerator, denominator


@lru_cache(maxsize=16)
def _permutation(height, width):
    seen, result, index = set(), [], 1
    while len(result) < height * width:
        xn, xd = radical_inverse(index, 2)
        yn, yd = radical_inverse(index, 3)
        position = (yn * height // yd) * width + xn * width // xd
        if position not in seen:
            seen.add(position)
            result.append(position)
        index += 1
    return tuple(result)


def halton_order(height, width, device=None):
    if min(height, width) < 1:
        raise ValueError("positive Halton grid required")
    return torch.tensor(_permutation(height, width), dtype=torch.long, device=device)
