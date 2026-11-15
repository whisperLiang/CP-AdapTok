import torch
from torch import nn
from torch.nn import functional as F


class CausalScorer(nn.Module):
    """Current ordered tokens, prefix budget and past-state summary only."""
    def __init__(self, dim=192, hidden=128):
        super().__init__()
        self.project = nn.Linear(dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features, budgets, history_mse=None):
        cumulative = self.project(features).cumsum(1)
        b, n, _ = cumulative.shape
        indices = torch.as_tensor(budgets, device=features.device, dtype=torch.long)
        if (indices < 1).any() or (indices > n).any():
            raise ValueError("invalid scorer budgets")
        pooled = cumulative[:, indices - 1] / indices[None, :, None]
        k = (indices.float() / n).expand(b, -1)
        past = torch.zeros(b, device=features.device) if history_mse is None else history_mse
        side = torch.stack((k, k.clamp_min(1e-6).log(), past[:, None].expand_as(k)), -1)
        return F.softplus(self.head(torch.cat((pooled, side), -1)).squeeze(-1))
