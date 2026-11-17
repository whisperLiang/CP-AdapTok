"""Spherical VQ with position-owned codebooks and explicit per-token loss reduction."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def code_statistics(ids, size=8192):
    counts = torch.bincount(ids.detach().flatten(), minlength=size).float()
    probabilities = counts / counts.sum().clamp_min(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum()
    return {"used_codes": int((counts > 0).sum()), "usage": float((counts > 0).float().mean()),
            "perplexity": float(entropy.exp()), "maximum_code_fraction": float(probabilities.max())}


@dataclass
class Quantized:
    ids: torch.Tensor
    quantized: torch.Tensor
    codebook_error: torch.Tensor
    commitment_error: torch.Tensor
    stats: dict
    loss_scope: str
    beta: float
    continuous: torch.Tensor

    def loss_terms(self, budget):
        if type(budget) is not int or not 1 <= budget <= self.ids.shape[1]:
            raise ValueError("VQ loss budget outside candidate sequence")
        count = budget if self.loss_scope == "active_prefix" else self.ids.shape[1]
        codebook = self.codebook_error[:, :count].mean()
        commitment = self.commitment_error[:, :count].mean()
        return {"codebook_loss": codebook, "commitment_loss": commitment,
                "vq_loss": codebook + self.beta * commitment}

    @property
    def loss(self):
        """Full-candidate loss for callers without a reconstruction budget."""
        return self.loss_terms(self.ids.shape[1])["vq_loss"]


class VectorQuantizer(nn.Module):
    def __init__(self, size=8192, dim=16, *, codebook_mode="shared", loss_scope="all_candidates",
                 counts=None, beta=.25):
        super().__init__()
        if codebook_mode not in ("shared", "per_level") or loss_scope not in ("all_candidates", "active_prefix"):
            raise ValueError("unknown VQ mode or loss scope")
        if codebook_mode == "per_level" and (counts is None or len(counts) != 3 or min(counts) < 1):
            raise ValueError("per-level VQ requires three positive position capacities")
        self.mode, self.loss_scope, self.counts, self.beta = codebook_mode, loss_scope, counts, beta
        self.codebook = self.new_book(size, dim)
        # Additional books must not move the RNG stream used to initialize the decoder.
        with torch.random.fork_rng(devices=[]):
            self.level_codebooks = nn.ModuleList(self.new_book(size, dim) for _ in range(2)
                                                 if codebook_mode == "per_level")

    @staticmethod
    def new_book(size, dim):
        book = nn.Embedding(size, dim)
        nn.init.normal_(book.weight)
        with torch.no_grad():
            book.weight.copy_(F.normalize(book.weight, dim=-1))
        return book

    @property
    def books(self):
        return (self.codebook, *self.level_codebooks)

    def identity_tensors(self):
        if self.mode == "shared":
            return {"codebook": self.codebook.weight}
        return {name: book.weight for name, book in zip(("Base", "Medium", "Fine"), self.books)}

    def segments(self, count):
        if self.mode == "shared":
            yield self.codebook, slice(0, count)
            return
        if count > sum(self.counts):
            raise ValueError("token count exceeds deterministic position table")
        offset = 0
        for capacity, book in zip(self.counts, self.books):
            end = min(count, offset + capacity)
            if end > offset:
                yield book, slice(offset, end)
            offset += capacity

    def lookup(self, ids):
        return torch.cat([F.embedding(ids[:, span], F.normalize(book.weight.float(), dim=-1))
                          for book, span in self.segments(ids.shape[1])], dim=1)

    def nearest(self, z):
        result = []
        for book, span in self.segments(z.shape[1]):
            codes = F.normalize(book.weight.detach().float(), dim=-1)
            values = z[:, span].detach()
            flat = values.reshape(-1, values.shape[-1])
            result.append(torch.cat([(chunk @ codes.T).argmax(dim=-1) for chunk in flat.split(512)])
                          .reshape(values.shape[:-1]))
        return torch.cat(result, dim=1)

    def forward(self, values):
        with torch.autocast(values.device.type, enabled=False):
            z = F.normalize(values.float(), dim=-1)
            with torch.no_grad():
                ids = self.nearest(z)
            discrete = self.lookup(ids)
            codebook = (discrete - z.detach()).square().mean(-1)
            commitment = (z - discrete.detach()).square().mean(-1)
            ste = discrete.detach() + (z - z.detach())
        stats = {**code_statistics(ids), "quantization_mse": float(commitment.detach().mean())}
        return Quantized(ids, ste, codebook, commitment, stats, self.loss_scope, self.beta, z)
