"""Fit scalar scales on Train and pick clipping on disjoint Train sources."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hla_tok.compression.packing import scalar_bounds
from hla_tok.compression.profiles import PROFILES_BY_LABEL, Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.compact import native_ranks
from hla_tok.streaming.continuous import MAX_K, NATIVE_RANKS, REGIONS, RankOrder
from scripts.hla_tok.compact.assets import load_asset, save_asset, transform_for
from scripts.hla_tok.compact.cache import load_train_tokens
from scripts.hla_tok.compact.inputs import TRAIN_PLAN
from scripts.hla_tok.compact.projection import asset_key
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.v2.protocol import json_at


def _rank_slices(rank_aware: bool) -> Iterator[tuple[int, slice]]:
    if rank_aware:
        for rank in NATIVE_RANKS:
            yield rank, slice(rank * REGIONS, min((rank + 1) * REGIONS, MAX_K))
    else:
        yield -1, slice(None)


def _clipping_candidates(fit: Tensor, percentiles: list[float], upper: int) -> Tensor:
    q = torch.tensor(percentiles, dtype=fit.dtype, device=fit.device).div_(100)
    width = max(1, (1 << 24) // len(fit))
    cuts = torch.cat([torch.quantile(block.abs(), q, dim=0) for block in fit.split(width, dim=1)], dim=1)
    return (cuts / upper).clamp_min(1e-8)


def _candidate_error(holdout: Tensor, scales: Tensor, variance: Tensor, low: int, high: int) -> Tensor:
    levels = (holdout / scales).round().clamp(low, high)
    return ((levels * scales - holdout).square() / variance).sum(dtype=torch.float64)


def _calibration_blocks(values: Tensor, fit_count: int, rank_aware: bool, bits: int,
                        percentiles: list[float]) -> tuple[list[Tensor], Tensor, int]:
    low, high = scalar_bounds(bits)
    totals = values.new_zeros(len(percentiles), dtype=torch.float64)
    out = []
    denominators = 0
    for _, slots in _rank_slices(rank_aware):
        d = values.shape[-1]
        fit = values[:fit_count, slots].reshape(-1, d).float()
        held = values[fit_count:, slots].reshape(-1, d).float()
        candidates = _clipping_candidates(fit, percentiles, high)
        normalizer = fit.var(0, unbiased=False).clamp_min(1e-8)
        totals += torch.stack([_candidate_error(held, scale, normalizer, low, high) for scale in candidates])
        denominators += held.numel()
        out.append(candidates)
    return out, totals, denominators


def _errors(held: Tensor, scale: Tensor, low: int, high: int) -> tuple[Tensor, np.ndarray]:
    scaled = held / scale
    q = scaled.round().clamp(low, high)
    result = torch.stack(((q * scale - held).square().sum(dim=0, dtype=torch.float64),
                          ((scaled < low) | (scaled > high)).sum(dim=0),
                          ((q == low) | (q == high)).sum(dim=0)))
    ids = q.to(torch.int32).cpu().numpy() - low
    counts = np.stack([np.bincount(channel, minlength=high - low + 1) for channel in ids.T])
    return result, counts


def _level_rows(counts: np.ndarray) -> list[dict]:
    records = []
    for i, levels in enumerate(counts):
        distribution = levels[levels > 0] / levels.sum()
        records.append({"channel": i, "levels_used": len(distribution),
                        "entropy_bits": float(-(distribution * np.log2(distribution)).sum()),
                        "level_histogram": " ".join(map(str, levels))})
    return records


def fit_scalar(values: Tensor, fit_count: int, rank_aware: bool, bits: int,
               percentiles: list[float]) -> tuple[StaticScalar, list[dict], list[dict]]:
    low, high = scalar_bounds(bits)
    candidates, sums, n = _calibration_blocks(values, fit_count, rank_aware, bits, percentiles)
    errors = sums.tolist()
    chosen = min(range(len(errors)), key=lambda i: (errors[i], -percentiles[i]))
    scales = torch.stack([table[chosen] for table in candidates])
    held = values[fit_count:]
    stats = [_errors(held[:, slots].reshape(-1, values.shape[-1]).float(), scale, low, high)
             for (_, slots), scale in zip(_rank_slices(rank_aware), scales, strict=True)]
    measures = torch.stack([item[0] for item in stats]).sum(dim=0).cpu().numpy()
    histograms = np.stack([item[1] for item in stats]).sum(axis=0)
    scalar_errors = measures.sum(1) / n
    level_details = _level_rows(histograms)
    for row, col in zip(level_details, measures.T, strict=True):
        row.update(dict(zip(("quantization_mse", "clipping_ratio", "saturation_ratio"),
                            (col / (n // len(level_details))).tolist(), strict=True)))
    records = [{"clip_percentile": p, "chosen": i == chosen, "holdout_normalized_mse": value / n}
               for i, (p, value) in enumerate(zip(percentiles, errors, strict=True))]
    records[chosen].update({"quantization_mse": float(scalar_errors[0]), "clipping_ratio": float(scalar_errors[1]),
                            "saturation_ratio": float(scalar_errors[2]),
                            "mean_channel_entropy_bits": float(np.mean([row["entropy_bits"] for row in level_details]))})
    return StaticScalar(scales, bits), records, level_details


@torch.inference_mode()
def quantized_assets(inputs: Inputs, config: dict, output: Path) -> dict:
    selected = json_at(output / "projection_selection.json")
    specs = json_at(output / "projection_assets.json")
    source = output / "rank_covariance.npz"
    if sha256_file(source) != json_at(output / "representation_audit.json")["covariance_sha256"]:
        raise ValueError("Train covariance identity changed")
    with np.load(source, allow_pickle=False) as archive:
        stats = dict(archive)
    cache = load_train_tokens(inputs, config, output)
    native = cache.native()
    ranks = native_ranks(RankOrder.native(), "cuda")
    order = RankOrder.reversed("cuda")
    model_id = inputs.checkpoint["state_dict_sha256"]
    calibrations: list[dict] = []
    levels: list[dict] = []
    for label in config["quantization"]["profiles"]:
        profile = PROFILES_BY_LABEL[label]
        if profile is Profile.RAW32:
            continue
        kind = "identity" if profile.dimension == 32 else selected[str(profile.dimension)]["projection"]
        key = asset_key(profile, kind)
        transform = transform_for(stats, kind, profile.dimension).cuda()
        with compute(config["compute_dtype"]):
            u = transform.encode(native, ranks)
        scalar, records, detail = fit_scalar(u, cache.fit_frames, kind == "rank", profile.bits,
                                             config["calibration"]["clip_percentiles"])
        calibrations.extend({"profile": label, "projection": kind, "split": "train_holdout"} | row for row in records)
        levels.extend({"profile": label, "projection": kind, "qmin": scalar.low} | row for row in detail)
        path = output / "profiles" / f"{key}.npz"
        spec = save_asset(path, profile, kind, transform, scalar)
        _, deployment = load_asset(path, spec, profile, order, model_id)
        specs[key] = spec | deployment | {"train_plan_sha256": sha256_file(TRAIN_PLAN),
                                          "clip_percentile": next(row["clip_percentile"] for row in records
                                                                  if row["chosen"])}
        print(f"Train scalar calibration: {key}", flush=True)
    write_csv(output / "scalar_calibration.csv", calibrations)
    write_csv(output / "scalar_integer_levels.csv", levels)
    write_json(output / "compact_profile_manifest.json", specs)
    return specs
