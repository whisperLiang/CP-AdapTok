"""Receiver-only FSQ synthesis and dual-mask residual recovery."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.c1.recovery_net import RecoveryNet as C1RecoveryNet
from aip_vtok.c1.scan import GROUP_WIDTH, GROUPS, gaussian_fill

B2 = "B2"
ZERO = "B2-ZeroFill"
B3 = "B3"


def require_prefixes(payload, prefixes):
    if not all(torch.equal(prefix, payload[:, :prefix.shape[1]]) for prefix in prefixes):
        raise RuntimeError("sparse prefix changed its integer FSQ IDs")


@dataclass(frozen=True)
class Observations:
    values: torch.Tensor  # [B, G, H, W, 6]
    mask_l1: torch.Tensor
    mask_l2: torch.Tensor


def sparse_values(quantizer, ids, masks):
    primary = masks[:, 0]
    refinement = masks[:, 1] if masks.shape[1] == 2 else torch.zeros_like(primary)
    if (refinement & ~primary).any():
        raise ValueError("Level 2 requires the corresponding Level 1")
    coarse = quantizer.primary.decode(ids[:, 0])
    if quantizer.residual is not None and masks.shape[1] == 2:
        delta = quantizer.residual.decode(ids[:, 1])
        coarse = torch.where(refinement[..., None], coarse + delta, coarse)
    elif refinement.any():
        raise ValueError("received Level 2 for single-stage FSQ")
    return Observations(torch.where(primary[..., None], coarse, 0), primary, refinement)


def interpolate(observations, sigma):
    planes = observations.values.movedim(-1, 2)
    return gaussian_fill(planes, observations.mask_l1, sigma).movedim(2, -1)


class RecoveryNet(C1RecoveryNet):
    """Keep the C1-A body and zero output heads, with eight arrival channels."""

    def __init__(self, width=48, context_channels=256):
        super().__init__(width, context_channels)
        self.inputs = nn.ModuleDict({
            key: nn.Conv2d(head.in_channels + GROUPS, width, 1) for key, head in self.inputs.items()
        })

    def forward(self, canvas, mask_l1, mask_l2, top_indices, causal_ctx, *, base):
        data = [canvas, mask_l1.to(canvas), mask_l2.to(canvas)]
        key = "I" if causal_ctx.intra else "P"
        if not causal_ctx.intra:
            data.append(F.avg_pool2d(causal_ctx.context.detach(), 2))
        x = torch.cat(data, dim=1).contiguous(memory_format=torch.channels_last)
        correction = self.outputs[key](self.body(self.inputs[key](x))).to(base.dtype)
        observed = torch.zeros_like(base, dtype=torch.bool)
        observed[:, top_indices] = mask_l1.repeat_interleave(GROUP_WIDTH, dim=1)
        return torch.where(observed, canvas, base + correction)


@dataclass(frozen=True)
class Receiver:
    sigma: float | None
    model: RecoveryNet | None = None

    def __call__(self, frame, backbone, context, payload):
        layout = frame.layout
        obs = frame.plan.receive(layout.quantizer, payload)
        values = obs.values if self.sigma is None else interpolate(obs, self.sigma)
        base = layout.restore(values, backbone)
        if self.model is None:
            return base
        canvas = layout.restore(obs.values, backbone)
        return self.model(canvas, obs.mask_l1, obs.mask_l2, layout.indices, context, base=base)
