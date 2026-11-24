"""Shared deterministic space/rank identities for local query tokenization."""

from hashlib import sha256

import torch
from torch import Tensor

from hla_tok.progressive.geometry import FEATURE_SIDE, MAX_QUERIES

GRID_CELLS = FEATURE_SIDE**2
FIRST_RANKS = 16
DIRECT_CAPACITY = 2 * GRID_CELLS


def grid_positions() -> Tensor:
    axis = torch.arange(FEATURE_SIDE)
    return torch.cartesian_prod(axis, axis)[:, [1, 0]]


def anchor_table() -> Tensor:
    """Four-by-four coverage, followed by greedy squared-distance coverage.

    Ties select the smallest row-major grid index. Each pass repeats that fixed
    spatial order so each rank always has one public location.
    """
    grid = grid_positions()
    selected = [y * FEATURE_SIDE + x for y in (0, 2, 5, 7) for x in (0, 2, 5, 7)]
    distances = (grid[:, None] - grid[None]).square().sum(-1)
    nearest = distances[:, selected].amin(1)
    while len(selected) < GRID_CELLS:
        next_index = int(nearest.argmax())
        selected.append(next_index)
        nearest = torch.minimum(nearest, distances[:, next_index])
    first = grid[selected]
    return first.repeat(MAX_QUERIES // GRID_CELLS, 1)


def gaussian_bias(table: Tensor) -> Tensor:
    sigma = torch.full((len(table),), 0.8, device=table.device)
    sigma[:FIRST_RANKS] = 2.0
    sigma[FIRST_RANKS:GRID_CELLS] = 1.25
    grid = grid_positions().to(table.device)
    squared = (table[:, None] - grid[None]).square().sum(-1)
    return -squared / (2 * sigma[:, None].square())


def position_embedding(table: Tensor, width: int) -> Tensor:
    frequency = torch.logspace(-1, 0, width // 4)
    angle = table.float().unsqueeze(-1) * frequency
    return torch.cat((angle.sin(), angle.cos()), dim=-1).flatten(1) * 0.02


def anchor_digest(table: Tensor) -> str:
    return sha256(table.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()
