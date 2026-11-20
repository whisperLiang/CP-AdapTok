"""A2 integer digit semantics with learnable scales and an FP32 surrogate."""

from dataclasses import dataclass

import torch
from torch import nn

from aip_vtok.c1_followup.quantization import CARDINALITY, ORIGINAL

from .transform import frame_type

GROUPS = 8
GROUP_WIDTH = len(ORIGINAL)
SENT_CHANNELS = GROUPS * GROUP_WIDTH


@dataclass(frozen=True)
class Quantized:
    ids: torch.Tensor
    values: torch.Tensor
    digit_position: torch.Tensor
    saturated: torch.Tensor


class ProgressiveFSQ(nn.Module):
    def __init__(self, levels: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        if levels.shape != (2, GROUPS, GROUP_WIDTH) or not torch.all(
            (levels >= 2).all(-1) & (levels.prod(-1) == CARDINALITY)
        ):
            raise ValueError("each FSQ group must have six dimensions and 64000 symbols")
        if scales.shape != levels.shape or not (scales.isfinite() & (scales > 0)).all():
            raise ValueError("expected finite positive A2 scales")
        self.register_buffer("levels", levels.long())
        self.register_buffer("radix", torch.cat((torch.ones_like(levels[..., :1]), levels.cumprod(-1)[..., :-1]), -1))
        self.register_buffer("initial_scale", scales.float())
        self.log_scale = nn.Parameter(torch.zeros_like(self.initial_scale))

    def scales(self, intra: bool) -> torch.Tensor:
        kind = frame_type(intra)
        return self.initial_scale[kind] * self.log_scale[kind].float().clamp(-6, 6).exp()

    def _coordinates(self, values: torch.Tensor, intra: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 5 or (values.shape[1], values.shape[-1]) != (GROUPS, GROUP_WIDTH):
            raise ValueError("expected [B,8,H,W,6] coefficients")
        levels = self.levels[frame_type(intra), :, None, None, :]
        source = values.float() / self.scales(intra)[:, None, None, :]
        if not source.isfinite().all():
            raise ValueError("FSQ received nonfinite normalized coefficients")
        extent, offset, shift = bin_parameters(levels)
        activated = torch.tanh(source + shift)
        bounded = activated * extent - offset
        return bounded, levels // 2, 1 - activated.detach().square() < 0.05

    def _ids(self, bounded: torch.Tensor, half: torch.Tensor, intra: bool) -> torch.Tensor:
        digits = bounded.detach().round().long() + half
        return (digits * self.radix[frame_type(intra), :, None, None, :]).sum(-1)

    @torch.no_grad()
    def encode_ids(self, values: torch.Tensor, intra: bool) -> torch.Tensor:
        bounded, half, _ = self._coordinates(values, intra)
        return self._ids(bounded, half, intra)

    def encode_train(self, values: torch.Tensor, intra: bool) -> Quantized:
        bounded, half, saturated = self._coordinates(values, intra)
        ids = self._ids(bounded, half, intra)
        levels = self.levels[frame_type(intra), :, None, None, :]
        digits = ids[..., None] // self.radix[frame_type(intra), :, None, None, :] % levels
        scale = self.scales(intra)[:, None, None, :]
        hard = decode_centers(digits, levels) * scale
        proxy = bounded / half * scale.detach()
        return Quantized(ids, hard + (proxy - proxy.detach()), bounded + half, saturated)

    def digits(self, ids: torch.Tensor, groups: torch.Tensor, intra: bool) -> torch.Tensor:
        if ids.dtype != torch.long or ids.ndim != 2 or groups.shape != (ids.shape[1],):
            raise ValueError("expected int64 IDs [B,N] and N group indices")
        if ((ids < 0) | (ids >= CARDINALITY)).any():
            raise ValueError("FSQ IDs must be in [0,64000)")
        kind = frame_type(intra)
        return ids[..., None] // self.radix[kind, groups] % self.levels[kind, groups]

    def decode_ids(self, ids: torch.Tensor, groups: torch.Tensor, intra: bool) -> torch.Tensor:
        levels = self.levels[frame_type(intra), groups]
        return decode_centers(self.digits(ids, groups, intra), levels) * self.scales(intra)[groups]


def bin_parameters(levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extent = (levels.float() - 1) / 2
    offset = (levels % 2 == 0).float() / 2
    return extent, offset, torch.atanh(offset / extent)


def decode_centers(digits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Invert the inherited A2 tanh centers, with finite outer bin midpoints."""
    extent, offset, shift = bin_parameters(levels)
    half = levels // 2
    codes = (digits - half).float() / half
    normalized = (codes * half + offset) / extent
    bound = 1 - 0.25 / extent
    return torch.atanh(normalized.clamp(-bound, bound)) - shift
