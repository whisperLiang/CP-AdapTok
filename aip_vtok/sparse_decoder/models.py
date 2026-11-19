"""Receiver architectures that share the frozen observation and history contract."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.backbones.latent_to_ref import DepthwiseResidual
from aip_vtok.c1.channels import CHANNELS
from aip_vtok.c1.scan import BUDGETS, GROUP_WIDTH, GROUPS
from aip_vtok.sparse_rfsq.protocol import SEED
from aip_vtok.sparse_rfsq.receiver import RecoveryNet, interpolate

ARMS = {"E2": "R48-G", "E3": "R-Wide", "E4": "SLD"}
CONTEXT_CHANNELS = 256
BUDGET_CHANNELS = 8


def preserve(canvas, base, correction, primary, indices):
    observed = torch.zeros_like(base, dtype=torch.bool)
    observed[:, indices] = primary.repeat_interleave(GROUP_WIDTH, dim=1)
    return torch.where(observed, canvas, base + correction.to(base.dtype))


def frame_input(data, context):
    if not context.intra:
        data.append(F.avg_pool2d(context.context.detach(), 2))
    return torch.cat(data, dim=1).contiguous(memory_format=torch.channels_last)


class R48G(RecoveryNet):
    """Retain W1's trunk and heads, with an input projection for the aligned Gaussian base."""

    def __init__(self):
        super().__init__()
        for kind, old in self.inputs.items():
            stem = nn.Conv2d(old.in_channels + CHANNELS, old.out_channels, 1)
            with torch.no_grad():
                stem.weight[:, :old.in_channels].copy_(old.weight)
                stem.bias.copy_(old.bias)
            self.inputs[kind] = stem

    def forward(self, canvas, primary, refinement, indices, context, *, base, budget):
        key = "I" if context.intra else "P"
        previous = frame_input([canvas, primary.to(canvas), refinement.to(canvas)], context)
        data = torch.cat((previous, base), dim=1).contiguous(memory_format=torch.channels_last)
        correction = self.outputs[key](self.body(self.inputs[key](data)))
        return preserve(canvas, base, correction, primary, indices)


class BudgetRecovery(nn.Module):
    """Common 24D observation, masks, budget, and I/P conditioning."""

    def __init__(self, width, body):
        super().__init__()
        self.budget_embedding = nn.Embedding(len(BUDGETS), BUDGET_CHANNELS)
        channels = 2 * GROUPS * (GROUP_WIDTH + 1) + BUDGET_CHANNELS
        self.inputs = nn.ModuleDict({"I": nn.Conv2d(channels, width, 1),
                                     "P": nn.Conv2d(channels + CONTEXT_CHANNELS, width, 1)})
        self.body = body
        self.outputs = nn.ModuleDict({kind: nn.Conv2d(width, CHANNELS, 1) for kind in self.inputs})
        for head in self.outputs.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def hidden(self, canvas, base, primary, refinement, indices, context, budget):
        key = "I" if context.intra else "P"
        embedding = self.budget_embedding.weight[BUDGETS.index(budget)].to(canvas)
        embedding = embedding.view(1, -1, 1, 1).expand(canvas.shape[0], -1, *canvas.shape[-2:])
        inputs = [canvas.index_select(1, indices), base.index_select(1, indices),
                  primary.to(canvas), refinement.to(canvas), embedding]
        return self.body(self.inputs[key](frame_input(inputs, context))), key

    def correction(self, hidden, key):
        return self.outputs[key](hidden)

    def forward(self, canvas, primary, refinement, indices, context, *, base, budget):
        hidden, key = self.hidden(canvas, base, primary, refinement, indices, context, budget)
        return preserve(canvas, base, self.correction(hidden, key), primary, indices)


class RWide(BudgetRecovery):
    def __init__(self, width=88):
        super().__init__(width, nn.Sequential(nn.SiLU(), *(DepthwiseResidual(width) for _ in range(4))))


class MultiscaleBody(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stem = nn.Sequential(nn.SiLU(), DepthwiseResidual(width))
        self.full = nn.Sequential(DepthwiseResidual(width), DepthwiseResidual(width))
        self.coarse = nn.Sequential(*(DepthwiseResidual(width) for _ in range(3)))
        self.fusion = nn.Sequential(nn.Conv2d(width * 2, width, 1), nn.SiLU(), DepthwiseResidual(width))

    def forward(self, x):
        stem = self.stem(x)
        full = self.full(stem)
        coarse = self.coarse(F.avg_pool2d(stem, 2))
        up = F.interpolate(coarse, size=full.shape[-2:], mode="nearest")
        return full + self.fusion(torch.cat((full, up), dim=1))


class SLD(BudgetRecovery):
    def __init__(self, width=64):
        super().__init__(width, MultiscaleBody(width))
        self.confidence = nn.Conv2d(width, CHANNELS, 1)

    def correction(self, hidden, key):
        return self.outputs[key](hidden) * self.confidence(hidden).sigmoid()


def create_model(arm, device="cpu"):
    torch.manual_seed(SEED)
    return {"E2": R48G, "E3": RWide, "E4": SLD}[arm]().to(device=device, memory_format=torch.channels_last)


@dataclass(frozen=True)
class DecoderReceiver:
    sigma: float
    model: nn.Module

    def __call__(self, frame, backbone, context, payload):
        layout = frame.layout
        obs = frame.plan.receive(layout.quantizer, payload)
        base = layout.restore(interpolate(obs, self.sigma), backbone)
        canvas = layout.restore(obs.values, backbone)
        return self.model(canvas, obs.mask_l1, obs.mask_l2, layout.indices, context,
                          base=base, budget=payload.shape[1])
