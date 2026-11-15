"""Deterministic VQ with AdapTok's normalized nearest-neighbor/straight-through losses."""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQResult:
    ids: torch.Tensor
    embeddings: torch.Tensor
    mask: torch.Tensor
    loss: torch.Tensor
    commitment: torch.Tensor
    codebook_loss: torch.Tensor
    usage: torch.Tensor
    perplexity: torch.Tensor


class AdapTokVQBottleneck(nn.Module):
    def __init__(
        self,
        input_dim=192,
        codebook_size=8192,
        embedding_dim=16,
        commitment=0.25,
        normalized=True,
        chunk_size=512,
    ):
        super().__init__()
        self.project_in = nn.Linear(input_dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, input_dim)
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.normal_(self.codebook.weight, std=embedding_dim**-0.5)
        self.normalized, self.beta, self.chunk_size = normalized, commitment, chunk_size

    def codebook_vectors(self):
        weight = self.codebook.weight.float()
        return F.normalize(weight, dim=-1) if self.normalized else weight

    def codebook_id(self):
        raw = self.codebook_vectors().detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()[:32]

    def embed(self, ids):
        if ids.dtype != torch.long or ids.ndim != 2 or ids.numel() == 0:
            raise ValueError("indices must be a nonempty [B,K] int64 tensor")
        if ids.min() < 0 or ids.max() >= self.codebook.num_embeddings:
            raise ValueError("codebook index out of range")
        return self.project_out(F.embedding(ids, self.codebook_vectors()))

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = self.project_in(features.float())
            z = F.normalize(z, dim=-1) if self.normalized else z
            weight = self.codebook_vectors()
            flat = z.flatten(0, 1)
            indices = []
            for chunk in flat.split(self.chunk_size):
                distance = chunk.square().sum(-1, keepdim=True) + weight.square().sum(-1)
                distance = distance - 2 * chunk @ weight.T
                indices.append(distance.argmin(-1))
            ids = torch.cat(indices).view(z.shape[:2])
            quantized = F.embedding(ids, weight)
            commitment = F.mse_loss(z, quantized.detach())
            codebook_loss = F.mse_loss(quantized, z.detach())
            straight_through = z + (quantized - z).detach()
            counts = torch.bincount(ids.flatten(), minlength=len(weight)).float()
            probabilities = counts / counts.sum()
            perplexity = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
            return VQResult(
                ids,
                self.project_out(straight_through),
                torch.ones_like(ids, dtype=torch.bool),
                self.beta * commitment + codebook_loss,
                commitment,
                codebook_loss,
                (counts > 0).float().mean(),
                perplexity,
            )
