from types import SimpleNamespace

import torch

from experiments.fixed2048.backbone_control import continuous_intra_latent


def test_continuous_intra_control_retains_both_official_latent_scales():
    adapter = SimpleNamespace(quant=lambda name, qp: torch.tensor(2.0 if name == "q_scale_y_enc" else 3.0))
    latent = torch.tensor([1.0, -4.0])
    assert torch.equal(continuous_intra_latent(adapter, latent, 32), latent * 6)
