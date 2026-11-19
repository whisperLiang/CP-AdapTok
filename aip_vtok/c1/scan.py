"""Content-independent C1 slot order on the native 1/16 grid."""

import math
from functools import lru_cache

import torch
from torch.nn import functional as F

from aip_vtok.diagnostics.diag0.sparse import BUDGETS
from aip_vtok.diagnostics.diag0.sparse import REF_BUDGET as REF_BUDGET
from aip_vtok.tokenizer.halton import radical_inverse

SHAPE = (68, 120)
GROUPS = 4
GROUP_WIDTH = 6
TOP = GROUPS * GROUP_WIDTH
DENSE = GROUPS * math.prod(SHAPE)
SCHEDULES = ("S1", "S2a", "S2b", "S3")
SIGMAS = (1.5, 2.5, 4.0)
TARGET = (4096, 2048, 1229, 819)


@lru_cache(maxsize=16)
def halton_positions(height, width, attempts=None):
    """Try twice the grid size in base 2/3, then fill unseen cells in row-major order."""
    area = height * width
    attempts = 2 * area if attempts is None else attempts
    seen = {}
    for index in range(1, attempts + 1):
        xn, xd = radical_inverse(index, 2)
        yn, yd = radical_inverse(index, 3)
        seen[(yn * height // yd) * width + xn * width // xd] = None
    return (*seen, *(i for i in range(area) if i not in seen))


def interleave(weights, counts, area, length):
    """Choose the greatest accumulated quota deficit. Group index breaks ties."""
    used = [0] * GROUPS
    total = sum(weights)
    for tick in range(1, length + 1):
        candidates = (g for g in range(GROUPS) if counts[g] < area)
        group = max(candidates, key=lambda g: tick * weights[g] - total * used[g])
        yield group, counts[group]
        used[group] += 1
        counts[group] += 1


def schedule_groups(name):
    if name == "S2a":
        return (0,) * 510, TARGET
    if name == "S2b":
        return (0, 1) * 255, TARGET
    if name != "S3":
        raise ValueError(f"unknown schedule {name}")
    return (), None


@lru_cache(maxsize=64)
def scan_schedule(name, *, weights=(1, 1, 1, 1), budget=BUDGETS[-1]):
    """A slot number is exactly group * H * W + y * W + x."""
    if not 0 <= budget <= BUDGETS[-1]:
        raise ValueError("sparse budget outside the C1 protocol")
    area = math.prod(SHAPE)
    positions = halton_positions(*SHAPE)
    if name == "S1":
        return tuple(i // area * area + positions[i % area] for i in range(budget))
    anchor, target = schedule_groups(name)
    counts, prefix = [0] * GROUPS, []
    for group in anchor[:budget]:
        prefix.append(group * area + positions[counts[group]])
        counts[group] += 1
    priorities = weights if name == "S3" else tuple(t - c for t, c in zip(target, counts))
    return (*prefix, *(g * area + positions[i] for g, i in interleave(
        priorities, counts, area, budget - len(prefix),
    )))


@lru_cache(maxsize=16)
def gaussian_kernel(sigma, device, dtype):
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    return kernel / kernel.sum()


def gaussian_fill(values, mask, sigma, eps=1e-8):
    """Independent normalized convolution for each group; original observations win."""
    observed = mask[:, :, None]
    planes = torch.cat((torch.where(observed, values, 0), observed.to(values)), dim=2)
    # Project each numerator and the matching group's denominator with the same kernel.
    h, w = mask.shape[-2:]
    kernel = gaussian_kernel(sigma, values.device, values.dtype)
    radius = len(kernel) // 2
    planes = F.conv2d(planes.reshape(-1, 1, h, w), kernel.view(1, 1, 1, -1), padding=(0, radius))
    planes = F.conv2d(planes, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    planes = planes.reshape(*values.shape[:2], values.shape[2] + 1, h, w)
    return torch.where(observed, values, planes[:, :, :-1] / (planes[:, :, -1:] + eps))
