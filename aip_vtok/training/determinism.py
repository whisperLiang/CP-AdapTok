"""One owner for the reproducible CUDA training policy."""

import os

import torch


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
