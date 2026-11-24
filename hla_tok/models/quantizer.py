"""Standard shared vector quantization with bounded FP32 nearest-neighbor search."""

import torch
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    def __init__(self, size: int = 8192, dim: int = 32, vectors_per_block: int = 4096, codes_per_block: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        nn.init.normal_(self.embedding.weight, std=dim**-0.5)
        self.vectors_per_block = vectors_per_block
        self.codes_per_block = codes_per_block

    def lookup(self, ids: Tensor) -> Tensor:
        return self.embedding(ids)

    def _search_block(self, vectors: Tensor, codes: Tensor, norms: Tensor) -> Tensor:
        best, ids = None, None
        for start in range(0, len(codes), self.codes_per_block):
            end = start + self.codes_per_block
            # The vector norm is constant across all candidate words.
            distances = torch.addmm(norms[start:end], vectors, codes[start:end].T, alpha=-2.0)
            distance, index = distances.min(dim=-1)
            index = index + start
            if best is None:
                best, ids = distance, index
            else:
                better = distance < best
                best = torch.where(better, distance, best)
                ids = torch.where(better, index, ids)
        return ids

    @torch.no_grad()
    def forward(self, z: Tensor) -> Tensor:
        with torch.autocast(device_type=z.device.type, enabled=False):
            vectors = z.reshape(-1, z.shape[-1]).float()
            codes = self.embedding.weight.float()
            norms = codes.square().sum(dim=-1)
            blocks = vectors.split(self.vectors_per_block)
            ids = [self._search_block(block, codes, norms) for block in blocks]
            return torch.cat(ids).reshape(z.shape[:-1])
