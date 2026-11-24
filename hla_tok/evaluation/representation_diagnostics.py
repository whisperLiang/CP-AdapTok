"""FP32 representation statistics and read-only attention tracing."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from hla_tok.models.local import DIM, HEADS
from hla_tok.progressive.geometry import group_prefix

EPS = 1e-12


def distinct_cosine(values):
    unit = F.normalize(values.float(), dim=-1, eps=1e-6)
    count = unit.shape[-2]
    other = unit.sum(-2).square().sum(-1) - unit.square().sum((-1, -2))
    return other / max(1, count * (count - 1))


def centered_spectrum(centered):
    singular = torch.linalg.svdvals(centered.float())
    squared = singular.square()
    total = squared.sum(-1, keepdim=True)
    probs = squared / total.clamp_min(EPS)
    entropy = -(probs * probs.clamp_min(EPS).log()).sum(-1)
    live = total.squeeze(-1) > EPS
    effective = torch.where(live, entropy.exp(), 0.0)
    stable = torch.where(live, total.squeeze(-1) / squared[..., 0].clamp_min(EPS), 0.0)
    return singular, effective, stable


def representation_metrics(values):
    raw = values.float()
    common = raw.mean(-2, keepdim=True)
    centered = raw - common
    spectrum, effective, stable = centered_spectrum(centered)
    return {
        "raw_adjacent_cosine": F.cosine_similarity(raw[..., :-1, :], raw[..., 1:, :], dim=-1).mean(-1),
        "raw_distinct_cosine": distinct_cosine(raw),
        "centered_distinct_cosine": distinct_cosine(centered),
        "effective_rank": effective,
        "stable_rank": stable,
        "centered_variance": centered.square().mean((-1, -2)),
        "common_energy_fraction": common.square().mean((-1, -2)) / raw.square().mean((-1, -2)).clamp_min(EPS),
        "singular_values": spectrum,
    }


def summarize_prefix(values, regions):
    groups = group_prefix(values, regions)
    records = [representation_metrics(group.flatten(0, 1)) for group in groups]
    return {key: torch.cat([rec[key] for rec in records])
            for key in records[0]}


def explicit_attention(query, memory, bias=None):
    keys, values = memory
    scores = (query.float() @ keys.float().transpose(-2, -1)) / query.shape[-1]**0.5
    if bias is not None:
        scores = scores + bias.float()
    weights = scores.softmax(-1)
    return weights, weights @ values.float()


@contextmanager
def trace_cross_attention(query_encoder):
    """Observe actual layer inputs with hooks. Recheck the FP32 attention operation."""
    traces = [[] for _ in query_encoder.layers]
    maxima = [0.0 for _ in traces]

    def capture(layer_index):
        def pre_hook(layer, arguments):
            x, memory, *optional = arguments
            bias = optional[0] if optional else None
            batch, length, _ = x.shape
            query = layer.query(layer.query_norm(x)).reshape(batch, length, HEADS, DIM // HEADS).transpose(1, 2)
            weights, expected = explicit_attention(query, memory, bias)
            actual = F.scaled_dot_product_attention(query, *memory, attn_mask=bias)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            maxima[layer_index] = max(maxima[layer_index], (actual - expected).abs().max().item())
            traces[layer_index].append(weights)
        return pre_hook

    hooks = [layer.register_forward_pre_hook(capture(i)) for i, layer in enumerate(query_encoder.layers)]
    try:
        yield traces, maxima
    finally:
        for hook in hooks:
            hook.remove()


def attention_summary(weights, first, stop):
    """Weights have shape [region, head, absolute rank, 64 locations]."""
    values = weights[..., first:stop, :]
    coverage = values.mean(dim=(0, 2))
    axis = torch.arange(8, device=weights.device).float()
    xy = torch.cartesian_prod(axis, axis)[:, [1, 0]]
    centers = values @ xy
    entropy = -(values * values.clamp_min(EPS).log()).sum(-1)
    head_repeat = distinct_cosine(values.transpose(1, 2)).mean().item()
    rows = []
    for head in range(values.shape[1]):
        rows.append({"head": head, "start_rank": first, "stop_rank": stop,
                     "attention_entropy_nats": entropy[:, head].mean().item(),
                     "different_rank_attention_cosine": distinct_cosine(values[:, head]).mean().item(),
                     "different_head_attention_cosine": head_repeat,
                     "center_x": centers[:, head, :, 0].mean().item(),
                     "center_y": centers[:, head, :, 1].mean().item(),
                     "center_x_std": centers[:, head, :, 0].std(unbiased=False).item(),
                     "center_y_std": centers[:, head, :, 1].std(unbiased=False).item()})
    return rows, coverage
