"""Independent native-resolution causal VQ tokenizer experiments."""

import os

# Set before the first CUDA BLAS handle, including when the CLI is imported through -m.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .config import ModelConfig

__all__ = ["ModelConfig"]
