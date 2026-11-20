"""Frame-type-specific orthogonal coordinates in the aligned latent space."""

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 128


def frame_type(intra: bool) -> int:
    return 0 if intra else 1


class LatentTransform(nn.Module):
    def __init__(self, orders: torch.Tensor):
        super().__init__()
        expected = torch.arange(CHANNELS, device=orders.device)
        if orders.shape != (2, CHANNELS) or not torch.equal(orders.sort().values, expected.expand(2, -1)):
            raise ValueError("expected one channel permutation for each frame type")
        self.register_buffer("order", orders.long())
        self.register_buffer("inverse_order", orders.argsort(dim=-1))
        self.register_buffer("identity", torch.eye(CHANNELS), persistent=False)
        triangle = torch.triu_indices(CHANNELS, CHANNELS, offset=1)
        self.register_buffer("triangle", triangle, persistent=False)
        self.skew = nn.Parameter(torch.zeros(2, triangle.shape[-1]))

    def matrix(self, intra: bool) -> torch.Tensor:
        kind = frame_type(intra)
        with torch.autocast(device_type=self.skew.device.type, enabled=False):
            upper = torch.zeros_like(self.identity).index_put(tuple(self.triangle), self.skew[kind].float())
            skew = upper - upper.T
            # I - skew is nonsingular for every real skew-symmetric matrix.
            cayley = torch.linalg.solve_ex(self.identity - skew, self.identity + skew).result
            return cayley.index_select(1, self.inverse_order[kind])

    @staticmethod
    def project(latent: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return F.linear(latent.float().movedim(1, -1), matrix).movedim(-1, 1)
