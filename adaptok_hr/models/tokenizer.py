"""One shared deterministic position table, with a spatially distributed prefix."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def distributed_order(height, width):
    yy, xx = np.meshgrid((np.arange(height) + .5) / height,
                         (np.arange(width) + .5) / width, indexing="ij")
    points = np.stack((yy.ravel(), xx.ravel()), axis=1)
    distances = np.full(len(points), np.inf)
    current = int(np.square(points - .5).sum(axis=1).argmin())
    order = []
    for _ in range(len(points)):
        order.append(current)
        distances = np.minimum(distances, np.square(points - points[current]).sum(axis=1))
        distances[current] = -1
        current = int(distances.argmax())
    return torch.tensor(order, dtype=torch.long)


class PositionTable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shapes = config.grid_shapes
        positions = []
        self.counts = tuple(h * w for h, w in self.shapes)
        for level, (h, w) in enumerate(self.shapes):
            order = distributed_order(h, w)
            self.register_buffer(f"order_{level}", order)
            positions.append(torch.stack((torch.full_like(order, level), order // w, order % w), dim=1))
        self.register_buffer("positions", torch.cat(positions))

    def order(self, level):
        return getattr(self, f"order_{level}")

    def scatter(self, tokens):
        b, count, dim = tokens.shape
        maps, offset = [], 0
        for level, ((h, w), capacity) in enumerate(zip(self.shapes, self.counts)):
            sent = max(0, min(count - offset, capacity))
            indices = self.order(level)[:sent]
            values = tokens[:, offset:offset + sent].transpose(1, 2)
            grid = tokens.new_zeros(b, dim, capacity).scatter(2, indices[None, None].expand(b, dim, -1), values)
            mask = tokens.new_zeros(b, 1, capacity)
            mask[:, :, indices] = 1
            maps.append(torch.cat((grid, mask), dim=1).reshape(b, dim + 1, h, w))
            offset += capacity
        return tuple(maps)


class Tokenizer(nn.Module):
    def __init__(self, config, table):
        super().__init__()
        # The model owns the only table. Submodules reference it without registering a duplicate.
        object.__setattr__(self, "table", table)
        self.projections = nn.ModuleList(nn.Conv2d(c, config.embedding_dim, 1)
                                         for c in reversed(config.channels[4:]))

    def forward(self, features):
        outputs = []
        for level, (feature, projection) in enumerate(zip(features, self.projections)):
            values = projection(feature).flatten(2).transpose(1, 2)
            outputs.append(values[:, self.table.order(level)])
        return F.normalize(torch.cat(outputs, dim=1).float(), dim=-1)
