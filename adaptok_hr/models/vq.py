"""Shared spherical VQ. Forward values are exact codebook lookups, with STE gradients."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def code_statistics(ids, size=8192):
    counts = torch.bincount(ids.detach().flatten(), minlength=size).float()
    probabilities = counts / counts.sum().clamp_min(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum()
    return {"used_codes": int((counts > 0).sum()), "usage": float((counts > 0).float().mean()),
            "perplexity": float(entropy.exp())}


@dataclass
class Quantized:
    ids: torch.Tensor
    quantized: torch.Tensor
    loss: torch.Tensor
    stats: dict


class VectorQuantizer(nn.Module):
    def __init__(self, size=8192, dim=16):
        super().__init__()
        self.codebook = nn.Embedding(size, dim)
        nn.init.normal_(self.codebook.weight)
        with torch.no_grad():
            self.codebook.weight.copy_(F.normalize(self.codebook.weight, dim=-1))

    def lookup(self, ids):
        return F.embedding(ids, F.normalize(self.codebook.weight.float(), dim=-1))

    def forward(self, values):
        with torch.autocast(values.device.type, enabled=False):
            z = F.normalize(values.float(), dim=-1)
            codes = F.normalize(self.codebook.weight.float(), dim=-1)
            with torch.no_grad():
                flat = z.detach().reshape(-1, z.shape[-1])
                ids = torch.cat([(chunk @ codes.detach().T).argmax(dim=-1)
                                 for chunk in flat.split(512)]).reshape(z.shape[:-1])
            discrete = self.lookup(ids)
            loss = F.mse_loss(discrete, z.detach()) + .25 * F.mse_loss(z, discrete.detach())
            ste = discrete.detach() + (z - z.detach())
            error = F.mse_loss(z.detach(), discrete.detach())
        return Quantized(ids, ste, loss, {**code_statistics(ids), "quantization_mse": float(error)})
