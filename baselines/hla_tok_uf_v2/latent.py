"""Frozen D1 latent scales and the transmitted model identifier."""

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from baselines.hla_tok_uf_v2.refinement import RefinementQuantizer
from third_party.DCVC.src.models.image_model import g_ch_y as INTRA_CHANNELS
from third_party.DCVC.src.models.video_model_ld import g_ch_y as VIDEO_CHANNELS

LATENT_UNIT = 4
STRIDE = 8
MAX_APPLICATION_BYTES = 166666
BUDGETS = {5: 20833, 10: 41666, 20: 83333, 40: MAX_APPLICATION_BYTES}


@dataclass(frozen=True, slots=True)
class Segmentation:
    tile: int = 16
    channels: int = 12
    conditional: bool = True
    ranked: bool = True

def stream_tag(model: nn.Module, identifier: str) -> bytes:
    digest = hashlib.sha256(identifier.encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.digest()[:8]


def normalized_error(encoded_y: Tensor, decoded_y: Tensor, native_step: Tensor) -> Tensor:
    return (encoded_y.float() - decoded_y.float() / native_step.float()) / LATENT_UNIT

class UnifiedRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intra = RefinementQuantizer(INTRA_CHANNELS, initial_step=0.125)
        self.inter = RefinementQuantizer(VIDEO_CHANNELS, initial_step=0.125)
        self.log_scale = nn.Parameter(torch.tensor([0.25, -0.75]))

    def quantizer(self, intra: bool) -> RefinementQuantizer:
        return self.intra if intra else self.inter

    def scales(self) -> Tensor:
        return self.log_scale.float().clamp(-3, 3).exp()
