"""Count-preserving and rank-elision interventions for diagnostics."""

import torch
from torch import nn
from torch.profiler import record_function

from hla_tok.progressive.geometry import group_prefix, merge_regions

INTERVENTIONS = (
    "original", "regional_mean", "repeat_rank0", "late_rank_replacement",
    "shuffle_values", "perturb_rank_assignment", "delete_early16", "delete_middle16", "delete_late16",
)


def intervene(values, case, quantizer=None):
    length = values.shape[-2]
    ranks = torch.arange(length, device=values.device)
    if case == "regional_mean":
        common = values.float().mean(-2, keepdim=True)
        if quantizer is not None:
            common = quantizer.lookup(quantizer(common))
        values = common.expand_as(values)
    elif case == "repeat_rank0":
        values = values[..., :1, :].expand_as(values)
    elif case == "late_rank_replacement":
        start = 3 * length // 4
        replacement = torch.where(ranks >= start, ranks - start, ranks)
        values = values.index_select(-2, replacement)
    elif case == "shuffle_values":
        values = values.roll(length // 3, dims=-2)
    elif case == "perturb_rank_assignment":
        ranks = ranks.roll(length // 2)
    elif case.startswith("delete_"):
        start = {"delete_early16": 0, "delete_middle16": 48, "delete_late16": 104}[case]
        keep = (ranks < start) | (ranks >= start + 16)
        ranks, values = ranks[keep], values[..., keep, :]
    return values, ranks


def local_with_ranks(decoder, values, ranks):
    batch, regions, _, _ = values.shape
    projected = decoder.input(values.flatten(0, 1))
    context = projected + decoder.rank[ranks].to(projected.dtype)
    query = decoder.spatial.to(projected.dtype).expand(batch * regions, -1, -1)
    for layer in decoder.layers:
        query = layer(query, layer.prepare(context))
    return query.unflatten(0, (batch, regions))


def reconstruct_intervention(model, tokens, geometry, case, *, quantized=False):
    quantizer = model.quantizer if quantized else None
    packed = quantizer.lookup(tokens) if quantized else tokens
    local_outputs, count = [], 0
    for group in group_prefix(packed, geometry.num_regions):
        values, ranks = intervene(group, case, quantizer)
        count += group.shape[1] * len(ranks)
        local_outputs.append(local_with_ranks(model.local_decoder, values, ranks))
    grid = merge_regions(torch.cat(local_outputs, dim=1), geometry)
    return model.rgb_decoder(grid, geometry), count


class DenseOracle(nn.Module):
    """Read-only spatial encoder, trained bridge and shared RGB decoder topology."""

    def __init__(self, model, width):
        super().__init__()
        self.encoder = model.encoder.requires_grad_(False).eval()
        self.rgb_decoder = model.rgb_decoder
        self.width = width
        self.bridge = nn.Conv2d(128, 128, 1) if width == 128 else nn.Sequential(
            nn.Conv2d(128, width, 1), nn.Conv2d(width, 128, 1)
        )

    def forward(self, rgb, geometry):
        with torch.no_grad():
            _, coarse = self.encoder(rgb, geometry)
        with record_function("hla/dense_oracle"):
            return self.rgb_decoder(self.bridge(coarse), geometry)
