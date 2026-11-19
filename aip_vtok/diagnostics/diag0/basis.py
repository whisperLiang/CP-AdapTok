"""Exact streamed covariance PCA on aligned channels and nonoverlapping blocks."""

from dataclasses import dataclass

import torch

from aip_vtok.evaluation.tokenizer_oracle import unpack_patches


def latent_rows(latent, block=1):
    if block == 1:
        return latent.movedim(1, -1).reshape(-1, latent.shape[1])
    return unpack_patches(latent, block).flatten(0, 1)


def latent_grid(rows, shape, block=1):
    height, width = shape
    grid = rows.reshape(1, height // block, width // block, -1, block, block)
    return grid.permute(0, 3, 1, 4, 2, 5).reshape(1, -1, height, width).contiguous(
        memory_format=torch.channels_last
    )


def fp32_channel_matmul(samples, vectors):
    """Short dot products and a separate reduction bound FP32 roundoff at full rank."""
    partial = [left @ right for left, right in zip(samples.split(16, dim=1), vectors.split(16, dim=0))]
    return torch.stack(partial).sum(dim=0)


@dataclass(frozen=True)
class Basis:
    mean: torch.Tensor
    vectors: torch.Tensor
    eigenvalues: torch.Tensor
    diagonal: torch.Tensor
    observations: int

    def rotate(self, rows, vectors=None, *, centered=True):
        if centered:
            rows = rows - self.mean
        return rows @ (self.vectors if vectors is None else vectors)

    def restore(self, codes, vectors=None, *, centered=True):
        axes = self.vectors if vectors is None else vectors
        restored = codes @ axes.T
        return restored + self.mean if centered else restored

    def project(self, latent, rank, block=1):
        vectors = self.vectors[:, :rank]
        codes = self.rotate(latent_rows(latent, block), vectors)
        return latent_grid(self.restore(codes, vectors), latent.shape[-2:], block)

    def explained(self, rank):
        return float(self.eigenvalues[:rank].sum() / self.eigenvalues.sum())


def covariance(latents, block):
    count, total, gram = 0, 0, 0
    for latent in latents:
        samples = latent_rows(latent, block).double()
        count += len(samples)
        total = total + samples.sum(0)
        gram = gram + samples.T @ samples
    if count == 0:
        raise ValueError("PCA requires training observations")
    mean = total / count
    return (gram / count - torch.outer(mean, mean)).cpu(), mean, count


def fit_basis(latents, rank, block=1):
    cov, mean, count = covariance(latents, block)
    values, vectors = torch.linalg.eigh(cov)
    vectors = vectors[:, -rank:].flip(1)
    pivot = vectors.abs().argmax(dim=0, keepdim=True)
    vectors = vectors * vectors.gather(0, pivot).sign()
    return Basis(mean.float(), vectors.to(mean).float(), values.flip(0).clamp_min(0), cov.diag(), count)
