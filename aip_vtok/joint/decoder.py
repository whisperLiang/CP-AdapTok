"""Budget-conditioned SLD in transformed coordinates with hard observation preservation."""

import math
from bisect import bisect_left
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.c1.scan import gaussian_fill
from aip_vtok.decoder.causal_state import FrameContext
from aip_vtok.sparse_decoder.models import CONTEXT_CHANNELS, SLD, MultiscaleBody, frame_input

from .scan import BUDGETS
from .tokenizer import GROUP_WIDTH, GROUPS, SENT_CHANNELS
from .transform import CHANNELS

WIDTH = 64
EMBEDDING = 8
SIGMA = 1.5
BASE = 24


class JointSLD(nn.Module):
    def __init__(self):
        super().__init__()
        stem_channels = 2 * SENT_CHANNELS + GROUPS + EMBEDDING
        self.budget_embedding = nn.Embedding(len(BUDGETS), EMBEDDING)
        self.inputs = nn.ModuleDict({
            "I": nn.Conv2d(stem_channels, WIDTH, 1),
            "P": nn.Conv2d(stem_channels + CONTEXT_CHANNELS, WIDTH, 1),
        })
        self.body = MultiscaleBody(WIDTH)
        self.outputs = nn.ModuleDict({kind: nn.Conv2d(WIDTH, CHANNELS, 1) for kind in self.inputs})
        self.confidence = nn.Conv2d(WIDTH, CHANNELS, 1)

    def embedding(self, budget: int) -> torch.Tensor:
        upper = min(bisect_left(BUDGETS, budget), len(BUDGETS) - 1)
        weights = self.budget_embedding.weight
        if upper == 0 or BUDGETS[upper] == budget:
            return weights[upper]
        lo, hi = math.log2(BUDGETS[upper - 1]), math.log2(BUDGETS[upper])
        return torch.lerp(weights[upper - 1], weights[upper], (math.log2(budget) - lo) / (hi - lo))

    def forward(self, values: torch.Tensor, group_mask: torch.Tensor, context: FrameContext,
                budget: int, order: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = values.shape
        hard = group_mask.bool()
        with torch.autocast(device_type=values.device.type, enabled=False):
            filled = gaussian_fill(values.reshape(batch, GROUPS, GROUP_WIDTH, height, width), hard, SIGMA)
            filled = filled.reshape_as(values)
        embedding = self.embedding(budget).view(1, -1, 1, 1).expand(batch, -1, height, width)
        kind = "I" if context.intra else "P"
        hidden = self.body(self.inputs[kind](frame_input([values, filled, group_mask, embedding], context)))
        # The migrated confidence head remains shared in its original output order.
        confidence = self.confidence(hidden).index_select(1, order).sigmoid()
        correction = self.outputs[kind](hidden) * confidence
        predicted = F.pad(filled, (0, 0, 0, 0, 0, CHANNELS - SENT_CHANNELS)) + correction.float()
        first = torch.where(hard.repeat_interleave(GROUP_WIDTH, 1), values, predicted[:, :SENT_CHANNELS])
        return torch.cat((first, predicted[:, SENT_CHANNELS:]), 1)


def migrate_stem(old: torch.Tensor, new: torch.Tensor) -> None:
    width = old.shape[0]
    context = old.shape[1] - 64
    # Legacy: values24, fill24, mask4, refinement4, embedding8, context.
    old_columns = torch.tensor([*range(52), *range(56, 64 + context)])
    new_columns = torch.tensor([*range(24), *range(48, 72), *range(96, 100), *range(104, 112 + context)])
    if old_columns.numel() != new_columns.numel() or new.shape[:2] != (width, 112 + context):
        raise ValueError("legacy SLD input channel map does not match the checkpoint")
    new.zero_().index_copy_(1, new_columns, old.index_select(1, old_columns))


@torch.no_grad()
def migrate_sld(model: JointSLD, legacy: Mapping[str, torch.Tensor], orders: torch.Tensor) -> dict:
    """Copy parameter tensors in CPU coordinates and probe the shared observation domain."""
    state = model.state_dict()
    copied, reshaped, mismatches = [], {}, {}
    for name, target in state.items():
        old = legacy.get(name)
        if old is None:
            mismatches[name] = {"source": None, "destination": list(target.shape)}
        elif name == "budget_embedding.weight" and old.shape == (5, EMBEDDING):
            target[:len(old)].copy_(old)
            target[len(old):].copy_(old[-1])
            reshaped[name] = "K16320+ initialized from K8192"
        elif name in ("inputs.I.weight", "inputs.P.weight") and old.shape == (
            WIDTH, 64 + (CONTEXT_CHANNELS if ".P." in name else 0), 1, 1
        ):
            migrate_stem(old, target)
            reshaped[name] = "legacy observation, Gaussian fill, L1 mask, budget and context columns"
        elif target.shape != old.shape:
            mismatches[name] = {"source": list(old.shape), "destination": list(target.shape)}
        elif name.startswith("outputs."):
            kind = 0 if ".I." in name else 1
            target.copy_(old.index_select(0, orders[kind]))
            copied.append(name)
        else:
            target.copy_(old)
            copied.append(name)
    for name in legacy.keys() - state.keys():
        mismatches[name] = {"source": list(legacy[name].shape), "destination": None}
    comparison = None if mismatches else compare_initialization(model, legacy, orders)
    return {
        "migrated": copied, "adapted_shapes": reshaped, "shape_mismatches": mismatches,
        "new_parameters": ["stem enhancement observations, fill and mask slices", "three higher budget rows"],
        "reinitialized": ["new stem slices set to zero", *mismatches],
        "discarded": ["legacy level-2 mask columns, inactive in SP-1"],
        "output_mapping": "I/P permuted output rows and indexed shared confidence",
        "initialization_comparison": comparison,
    }


@torch.no_grad()
def compare_initialization(model: JointSLD, state: Mapping[str, torch.Tensor], orders: torch.Tensor) -> dict:
    old = SLD()
    old.load_state_dict(state)
    generator = torch.Generator().manual_seed(1)
    mask = torch.rand(1, 4, 8, 12, generator=generator) > 0.7
    values = torch.randn((1, BASE, 8, 12), generator=generator) * mask.repeat_interleave(6, 1)
    filled = gaussian_fill(values.reshape(1, 4, 6, 8, 12), mask, SIGMA).reshape_as(values)
    blank = torch.zeros((1, CHANNELS, 8, 12))
    results = {}
    for kind, order in enumerate(orders):
        p_context = None if kind == 0 else torch.randn(1, CONTEXT_CHANNELS, 16, 24, generator=generator)
        context = FrameContext(kind, kind == 0, p_context, None)
        indices = order[:BASE]
        canvas, base = blank.index_copy(1, indices, values), blank.index_copy(1, indices, filled)
        for budget in (510, 8192):
            expected = old(canvas, mask, torch.zeros_like(mask), indices, context, base=base, budget=budget)
            actual = model(F.pad(values, (0, 0, 0, 0, 0, BASE)), F.pad(mask.float(), (0, 0, 0, 0, 0, 4)),
                           context, budget, order)
            error = actual - expected[:, order]
            results[f'{"PI"[kind == 0]}_{budget}'] = {"max_abs": error.abs().max().item(),
                                                   "mse": error.square().mean().item()}
    return results
