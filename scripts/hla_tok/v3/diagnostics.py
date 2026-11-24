"""Checkpoint-only measurements for locating the B0 reconstruction gap."""

import json
from collections import defaultdict
from statistics import mean

import torch

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.bottleneck_diagnostics import reconstruct_intervention
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import summarize_quality
from hla_tok.evaluation.representation_diagnostics import summarize_prefix
from hla_tok.progressive.geometry import FEATURE_SIDE, REGION_SIZE

INTERVENTIONS = {
    "regional_mean": "token_content",
    "repeat_rank0": "token_content",
    "shuffle_values": "token_content",
    "perturb_rank_assignment": "rank_assignment",
    "delete_early16": "token_count",
    "delete_middle16": "token_count",
    "delete_late16": "token_count",
}
STATS = ("l1", "mse", "channel_normalized_rmse", "cosine_similarity")


def _spectrum(error):
    height, width = error.shape[-2:]
    power = torch.fft.rfft2(error, norm="ortho").abs().square().mean(dim=(0, 1))
    vertical = torch.fft.fftfreq(height, device=power.device)[:, None]
    horizontal = torch.fft.rfftfreq(width, device=power.device)[None, :]
    radius = (vertical.square() + horizontal.square()).sqrt()
    return {f"latent_error_power_{band}": power[region].mean().item() for band, region in
            (("low", radius < 0.1), ("mid", (radius >= 0.1) & (radius < 0.25)), ("high", radius >= 0.25))}


def _valid_axis(length, padded_cells, device):
    stride = REGION_SIZE // FEATURE_SIDE
    offset = torch.arange(padded_cells, device=device) * stride
    return (length - offset).clamp(0, stride) / stride


def _seam_axis(length, device):
    coord = torch.arange(length, device=device)
    return ((coord % FEATURE_SIDE == 0) & (coord > 0)) | (
        ((coord + 1) % FEATURE_SIDE == 0) & (coord + 1 < length))


def _spatial_weights(reference, geometry):
    height, width = reference.shape[-2:]
    device = reference.device
    valid = _valid_axis(geometry.height, height, device)[:, None] * _valid_axis(geometry.width, width, device)
    seam = _seam_axis(height, device)[:, None] | _seam_axis(width, device)
    return {"valid": valid, "padding": 1 - valid, "internal_seam": valid * seam, "interior": valid * ~seam}


def _weighted_metrics(reference, recovered, error, weights):
    weights = weights / weights.sum()
    axes = (-2, -1)

    def spatial_average(values, *, keepdim=False):
        return (values * weights).sum(dim=axes, keepdim=keepdim)

    ref_var = spatial_average((reference - spatial_average(reference, keepdim=True)).square())
    got_var = spatial_average((recovered - spatial_average(recovered, keepdim=True)).square())
    mse = spatial_average(error.square())
    cross = spatial_average(reference * recovered).sum(-1)
    norm = (spatial_average(reference.square()).sum(-1) * spatial_average(recovered.square()).sum(-1)).sqrt()
    values = torch.stack((spatial_average(error.abs()).mean(), mse.mean(),
                          (mse.sqrt() / ref_var.sqrt().clamp_min(1e-6)).mean(),
                          (cross / norm.clamp_min(1e-12)).mean()))
    return dict(zip(STATS, values.tolist(), strict=True)), torch.stack((ref_var, got_var)).mean(dim=1)


def latent_alignment(reference, recovered, geometry):
    """Weight coarse cells by their valid-pixel area, including the partial bottom cell."""
    reference, recovered = reference.float(), recovered.float()
    error = recovered - reference
    rows = [{"region": "whole", "l1": error.abs().mean().item(), "mse": error.square().mean().item()}
            | _spectrum(error)]
    variance_rows = []
    for region, weights in _spatial_weights(reference, geometry).items():
        if not weights.any():
            continue
        metrics, variances = _weighted_metrics(reference, recovered, error, weights)
        rows.append({"region": region} | metrics)
        if region in ("valid", "padding"):
            variance_rows.extend(
                {"region": region, "channel": channel, "reference_variance": left, "recovered_variance": right}
                for channel, (left, right) in enumerate(variances.T.tolist()))
    return rows, variance_rows


def _encode_with_dense_bypass(codec, source, budget):
    captured = []

    def capture(_encoder, _input, output):
        captured.append(output[1])

    with codec.encoder.register_forward_hook(capture):
        tokens = codec.encode_training(source, budget, quantize=False)
    coarse, = captured
    return tokens, coarse


def _reconstruction_paths(dense, codec, source, budget):
    with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=source.is_cuda):
        teacher, geometry = dense.coarse(source)
        tokens, coarse = _encode_with_dense_bypass(codec, source, budget)
        local = codec._decode_features(tokens.values, geometry)
        images = {
            "phase_a_dense": (dense.rgb_decoder(teacher, geometry), None),
            "b0_dense_through_b0_rgb": (codec.rgb_decoder(coarse, geometry), None),
            "b0_local": (codec.rgb_decoder(local, geometry), budget),
        }
        for case in INTERVENTIONS:
            images[case] = reconstruct_intervention(codec, tokens.values, geometry, case)
    pairs = {"phase_a_vs_b0_dense": (teacher, coarse), "b0_dense_vs_local": (coarse, local),
             "phase_a_vs_b0_local": (teacher, local)}
    return pairs, images, tokens.values, geometry


def _references(cohort, subset):
    for video, count in cohort.items():
        indices = (0, count // 2, count - 1) if subset else range(count)
        for index in indices:
            yield video, index


def _summaries(observations, subset):
    return [
        {"mode": label, "matched_174": not subset, "intervention": items[0]["intervention"],
         "received_tokens": items[0]["received_tokens"]} | summarize_quality(items)
        for label, items in observations.items()
    ]


def _write_pixel_reads(output, counts):
    path = output / "pixel_reads.json"
    previous = json.loads(path.read_bytes()) if path.is_file() else {}
    write_json(path, {key: previous.get(key, 0) + delta for key, delta in counts.items()})


def _add_latent_frame(pairs, geometry, keys, alignment, variances):
    for label, (reference, recovered) in pairs.items():
        stats, channel_rows = latent_alignment(reference, recovered, geometry)
        alignment.extend(keys | {"comparison": label} | row for row in stats)
        for row in channel_rows:
            variances[(label, row["region"], row["channel"])].append(row)


def _write_observations(output, observations, subset, alignment, variances, representation):
    rows = _summaries(observations, subset)
    write_csv(output / "latent_alignment_metrics.csv", alignment)
    write_csv(output / "latent_channel_variance.csv", [
        {"comparison": label, "region": region, "channel": channel, "frames": len(items)}
        | {metric: mean(item[metric] for item in items) for metric in ("reference_variance", "recovered_variance")}
        for (label, region, channel), items in variances.items()])
    write_csv(output / "token_representation.csv", representation)
    write_csv(output / "per_frame.csv", [row for items in observations.values() for row in items])
    write_csv(output / "metrics.csv", rows)
    return rows


@torch.inference_mode()
def diagnose(dense, codec, reader, cohort, quality, budget, output, *, subset=False):
    dense.eval()
    codec.eval()
    references = tuple(_references(cohort, subset))
    if len(set(references)) != (9 if subset else 174):
        raise ValueError("diagnostics require the frozen native trend or full cohort")

    before = dict(reader.counts)
    observations = defaultdict(list)
    alignment, variances, representation = [], defaultdict(list), []
    try:
        for video, index in references:
            source = transfer(reader.read("validation", video, index)[None])
            pairs, images, values, geometry = _reconstruction_paths(dense, codec, source, budget)
            keys = {"video": video, "frame": index}
            _add_latent_frame(pairs, geometry, keys, alignment, variances)
            stats = summarize_prefix(values, geometry.num_regions)
            representation.append(keys | {key: tensor.mean().item() for key, tensor in stats.items()
                                          if key != "singular_values"})
            for label, (image, count) in images.items():
                observations[label].append(keys | {"mode": label, "received_tokens": count,
                                                   "intervention": INTERVENTIONS.get(label)} | quality(source, image))
        return _write_observations(output, observations, subset, alignment, variances, representation)
    finally:
        _write_pixel_reads(output, {key: reader.counts[key] - before[key] for key in before})
