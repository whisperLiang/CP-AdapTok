"""Same-state distortion labels and strictly current/past online budget selection."""

import torch
from torch import nn
from torch.nn import functional as F


class BudgetScorer(nn.Module):
    def __init__(self, dim=16, budgets=(256, 512, 1024, 2048)):
        super().__init__()
        self.budgets = tuple(budgets)
        self.network = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, len(budgets)))

    def forward(self, current_features):
        return F.softplus(self.network(current_features))

    @torch.no_grad()
    def select(self, current_features, maximum_tokens=2048, rate_weight=0.0):
        costs = self(current_features)[0]
        budgets = torch.tensor(self.budgets, device=costs.device)
        costs = costs + rate_weight * budgets.float() * 13
        costs = costs.masked_fill(budgets > maximum_tokens, float("inf"))
        if not torch.isfinite(costs).any():
            raise ValueError("maximum_tokens below minimum supported budget")
        return self.budgets[int(costs.argmin())]


@torch.no_grad()
def same_state_labels(model, frame, state):
    model.eval().requires_grad_(False)
    encoded = model.encode_features(frame, state)
    labels = []
    for budget in model.config.budgets:
        rgb, _ = model.decode_ids(encoded.ids[:, :budget], state)
        labels.append((rgb.float() - frame.float()).square().flatten(1).mean(1))
    return encoded.quantized.mean(dim=1).detach(), torch.stack(labels, dim=1)


def scorer_update(model, scorer, optimizer, frame, state):
    features, labels = same_state_labels(model, frame, state)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(scorer(features), labels)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "labels": labels.cpu().tolist()}
