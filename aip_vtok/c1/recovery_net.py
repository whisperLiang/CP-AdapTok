"""Small masked residual recovery and a one-writer detached reference boundary."""

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.backbones.latent_to_ref import DepthwiseResidual

from .channels import CHANNELS
from .scan import GROUP_WIDTH, GROUPS


class RecoveryNet(nn.Module):
    """Operate at 1/16 resolution; only unobserved entries receive a correction."""

    def __init__(self, width=48, context_channels=256):
        super().__init__()
        self.inputs = nn.ModuleDict({
            "I": nn.Conv2d(CHANNELS + GROUPS, width, 1),
            "P": nn.Conv2d(CHANNELS + GROUPS + context_channels, width, 1),
        })
        self.body = nn.Sequential(nn.SiLU(), *(DepthwiseResidual(width) for _ in range(4)))
        self.outputs = nn.ModuleDict({key: nn.Conv2d(width, CHANNELS, 1) for key in self.inputs})
        for head in self.outputs.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, base, mask, top_indices, context):
        data = [base, mask.to(base)]
        key = "I" if context.intra else "P"
        if not context.intra:
            data.append(F.avg_pool2d(context.context.detach(), 2))
        hidden = self.body(self.inputs[key](torch.cat(data, dim=1).contiguous(memory_format=torch.channels_last)))
        correction = self.outputs[key](hidden).to(base.dtype)
        observed = torch.zeros_like(base, dtype=torch.bool)
        observed[:, top_indices] = mask.repeat_interleave(GROUP_WIDTH, dim=1)
        return torch.where(observed, base, base + correction)


def reference_supervision(backbone, projector, latent, native, context):
    with torch.no_grad():
        _, teacher = backbone.decode_native(native, context)
    feature = projector(latent, context.intra)
    return F.l1_loss(feature.float(), teacher.float()), feature


@torch.no_grad()
def commit_reference(context, latent, feature):
    return context.commit_reference_once(feature, latent)
