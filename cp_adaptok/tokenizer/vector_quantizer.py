"""One shared VQ; stochastic training and deterministic stream inference."""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQResult:
    ids: torch.Tensor | None
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
        self.strategy, self.temperature, self.sampling_topk = "deterministic_nearest", 1.0, 0
        self.register_buffer("usage_window", torch.zeros(0, codebook_size), persistent=False)
        self.register_buffer("pending_counts", torch.zeros(codebook_size), persistent=False)
        self.usage_updates = 0
        self.dead_min_count, self.dead_max_replacements = 1, 0
        self._candidates = None

    def set_training_policy(self, strategy, temperature=1.0, sampling_topk=0):
        if strategy not in ("continuous", "deterministic_nearest", "stochastic_training"):
            raise ValueError("unknown VQ training strategy")
        if temperature <= 0 or not 0 <= sampling_topk <= self.codebook.num_embeddings:
            raise ValueError("invalid VQ temperature/top-k")
        if strategy == "stochastic_training" and not self.normalized:
            raise ValueError("stochastic VQ requires normalized similarity")
        self.strategy, self.temperature, self.sampling_topk = strategy, temperature, sampling_topk

    def _select_indices(self, chunk, weight):
        if not self.training or self.strategy == "deterministic_nearest":
            return self.distances(chunk, weight).argmin(-1)
        similarities = chunk @ weight.T / self.temperature
        if self.sampling_topk:
            values, candidates = similarities.topk(self.sampling_topk, dim=-1)
            selected = torch.multinomial(values.softmax(-1), 1)
            return candidates.gather(-1, selected).squeeze(-1)
        return torch.multinomial(similarities.softmax(-1), 1).squeeze(-1)

    @torch.no_grad()
    def initialize_codebook(self, samples, seed, iterations=20):
        from .vq_initialization import kmeans_centers

        centers = kmeans_centers(samples, self.codebook.num_embeddings, seed, iterations, self.normalized)
        self.codebook.weight.copy_(centers.to(self.codebook.weight))

    def configure_dead_codes(self, window=0, min_count=1, max_replacements=0):
        if window < 0 or min_count < 1 or max_replacements < 0:
            raise ValueError("invalid dead-code window/count")
        self.usage_window = self.pending_counts.new_zeros(window, self.codebook.num_embeddings)
        self.usage_updates = 0
        self.pending_counts.zero_()
        self.dead_min_count, self.dead_max_replacements = min_count, max_replacements

    def training_state(self):
        return {"window": self.usage_window.cpu(), "updates": self.usage_updates}

    def restore_training_state(self, state):
        if state["window"].shape != self.usage_window.shape:
            raise ValueError("dead-code configuration differs from checkpoint")
        self.usage_window.copy_(state["window"])
        self.usage_updates = state["updates"]

    @torch.no_grad()
    def finish_update(self, optimizer):
        if not self.training:
            raise RuntimeError("dead-code recovery is training only")
        if not len(self.usage_window) or self._candidates is None:
            return []
        self.usage_window[self.usage_updates % len(self.usage_window)].copy_(self.pending_counts)
        self.pending_counts.zero_()
        self.usage_updates += 1
        dead = (self.usage_window.sum(0) < self.dead_min_count).nonzero().flatten()
        if self.usage_updates < len(self.usage_window):
            dead = dead[:0]
        dead = dead[: self.dead_max_replacements]
        if len(dead):
            choices = torch.randperm(len(self._candidates), device=dead.device)[: len(dead)]
            dead = dead[: len(choices)]
            self.codebook.weight[dead] = self._candidates[choices]
            for value in optimizer.state.get(self.codebook.weight, {}).values():
                if torch.is_tensor(value) and value.shape == self.codebook.weight.shape:
                    value[dead] = 0
            self.usage_window[:, dead] = self.dead_min_count
        self._candidates = None
        return dead.cpu().tolist()

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

    def project_features(self, features):
        z = self.project_in(features.float())
        return F.normalize(z, dim=-1) if self.normalized else z

    @staticmethod
    def distances(chunk, weight):
        return chunk.square().sum(-1, keepdim=True) + weight.square().sum(-1) - 2 * chunk @ weight.T

    def expected_bits(self, features, block_logits, packet_tokens, temperature=0.1):
        """Differentiable rate surrogate only; measured bitrate always comes from rANS bytes."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = self.project_features(features)
            weight = self.codebook_vectors()
            b, n, _ = z.shape
            block = torch.arange(n, device=z.device).repeat(b) // packet_tokens
            costs = -block_logits.float().log_softmax(-1) / torch.log(z.new_tensor(2.0))
            output = []
            for start in range(0, b * n, self.chunk_size):
                chunk = z.flatten(0, 1)[start : start + self.chunk_size]
                probabilities = (-self.distances(chunk, weight) / temperature).softmax(-1)
                output.append((probabilities * costs[block[start : start + len(chunk)]]).sum(-1))
            return torch.cat(output).view(b, n)

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = self.project_features(features)
            if self.strategy == "continuous":
                if not self.training:
                    raise RuntimeError("continuous diagnostic cannot produce a compression bitstream")
                zero = z.new_zeros(())
                return VQResult(
                    None,
                    self.project_out(z),
                    torch.ones(z.shape[:2], device=z.device, dtype=torch.bool),
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                )
            weight = self.codebook_vectors()
            flat = z.flatten(0, 1)
            indices = []
            for chunk in flat.split(self.chunk_size):
                indices.append(self._select_indices(chunk, weight))
            ids = torch.cat(indices).view(z.shape[:2])
            quantized = F.embedding(ids, weight)
            commitment = F.mse_loss(z, quantized.detach())
            codebook_loss = F.mse_loss(quantized, z.detach())
            straight_through = z + (quantized - z).detach()
            counts = torch.bincount(ids.flatten(), minlength=len(weight)).float()
            if self.training and len(self.usage_window):
                self.pending_counts.add_(counts.detach())
                candidates = flat.detach()
                self._candidates = (
                    candidates if self._candidates is None else torch.cat((self._candidates, candidates))
                )
                self._candidates = self._candidates[-16384:]
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
