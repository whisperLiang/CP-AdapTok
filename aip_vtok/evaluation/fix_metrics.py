"""Color and FSQ measurements for Gate 1-Fix."""

import numpy as np
import torch

from aip_vtok.training.losses import chroma_diagnostics


def fix_chroma(original, reconstruction):
    values = chroma_diagnostics(original, reconstruction)
    return {
        "chroma_var_retention_pct": 100 * values["chroma_retention"],
        "chroma_mse": values["chroma_mse"],
        "chroma_mean_bias": values["chroma_mean_abs_bias"],
    }


def chroma_passes(metrics):
    return (
        75 <= metrics["chroma_var_retention_pct"] <= 130
        and metrics["chroma_mse"] <= 0.0025
        and metrics["chroma_mean_bias"] <= 0.015
    )


def entropy_bits(counts):
    values = np.asarray(counts)
    p = values[values > 0] / values.sum()
    return float(-np.sum(p * np.log2(p)))


def fsq_stats(ids, dims, saturated):
    elements = sum(sum(row) for row in dims)
    boundary = sum(row[0] + row[-1] for row in dims)
    id_entropy = entropy_bits(ids)
    return {
        "tokens": int(elements // 6),
        "unique_ids": int(np.count_nonzero(ids)),
        "id_entropy_bits": id_entropy,
        "effective_ids": 2**id_entropy if elements else 0.0,
        "dim_entropy_bits": [entropy_bits(row) for row in dims],
        "boundary_code_ratio": boundary / max(elements, 1),
        "tanh_saturation_ratio": saturated / max(elements, 1),
    }


def frame_fsq(encoded, fsq):
    tokens = encoded.tokens
    if tokens.saturation is None:
        raise ValueError("FSQ health requires the actual pre-rounding tanh diagnostic")
    result = {}
    levels = fsq.levels.tolist()
    for pass_id in range(3):
        mask = tokens.plan.pass_ids == pass_id
        ids = tokens.ids[:, mask]
        values, counts = ids.unique(return_counts=True)
        digits = fsq.ids_to_digits(ids).flatten(0, -2).cpu()
        dims = [torch.bincount(digits[:, i], minlength=n).tolist() for i, n in enumerate(levels)]
        saturated = int(tokens.saturation[:, mask].sum())
        result[str(pass_id)] = {
            "id_counts": list(zip(values.tolist(), counts.tolist())),
            "dim_counts": dims,
            "saturated_elements": saturated,
            **fsq_stats(counts.tolist(), dims, saturated),
        }
    return result


def percentiles(values):
    return {
        key: np.quantile(values, quantile, axis=0).tolist()
        for key, quantile in (("min", 0), ("p05", 0.05), ("p95", 0.95), ("max", 1))
    }


def pooled_fsq(rows):
    result = {}
    for pass_id in ("0", "1", "2"):
        frames = [row["passes"][pass_id] for row in rows]
        ids = np.zeros(64000, dtype=np.int64)
        dims = [np.zeros(len(row), dtype=np.int64) for row in frames[0]["dim_counts"]]
        for frame in frames:
            for index, count in frame["id_counts"]:
                ids[index] += count
            for total, histogram in zip(dims, frame["dim_counts"]):
                total += histogram
        saturated = sum(frame["saturated_elements"] for frame in frames)
        result[pass_id] = {
            "pooled": fsq_stats(ids, [row.tolist() for row in dims], saturated),
            "per_frame": {
                key: percentiles([frame[key] for frame in frames])
                for key in (
                    "dim_entropy_bits",
                    "effective_ids",
                    "boundary_code_ratio",
                    "tanh_saturation_ratio",
                )
            },
        }
    return result
