"""File and tensor identities for reproducible bitstream checks."""

import hashlib
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor


def sha256_file(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def tensor_digest(tensors: list[Tensor]) -> str:
    digest = sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()
