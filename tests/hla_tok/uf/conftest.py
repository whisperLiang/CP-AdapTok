import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_assets():
    import torch

    from hla_tok.uf.official import OfficialAssets
    from scripts.hla_tok.v2.protocol import setup

    directory = os.environ.get("UF_OFFICIAL_WEIGHTS")
    if not directory or not torch.cuda.is_available():
        pytest.skip("native H100 test requires UF_OFFICIAL_WEIGHTS and CUDA")
    setup()
    return OfficialAssets.verify(Path(directory))
