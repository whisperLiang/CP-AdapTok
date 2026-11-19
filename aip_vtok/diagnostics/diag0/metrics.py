"""The sealed evaluation metrics with matched subgroup and FSQ pooling."""

from collections import defaultdict

import numpy as np
from torch.nn import functional as F

from aip_vtok.evaluation.fix_metrics import entropy_bits, fix_chroma
from aip_vtok.evaluation.gates import distortion_summary


def measure(suite, rgb, rec, latent, target):
    return {
        **suite(rgb, rec), **fix_chroma(rgb, rec),
        "latent_mse": F.mse_loss(latent, target).item(),
    }


def summarize(rows):
    summary = distortion_summary(rows)
    ignored = {"frame", "mse", "psnr", "budget", "rank", "block", "reference_budget"}
    for field, first in rows[0].items():
        if field in ignored or type(first) not in (int, float):
            continue
        summary[field] = sum(row[field] for row in rows) / len(rows)
    return summary


def fsq_summary(counts, saturated):
    groups = []
    for group_id, (hist, sat) in enumerate(zip(counts, saturated)):
        tokens = int(hist[0].sum())
        groups.append({
            "group": group_id if group_id < 21 else "anchor",
            "tokens": tokens,
            "tanh_saturation_ratio": int(sat) / max(6 * tokens, 1),
            "dim_entropy_bits": [entropy_bits(row) for row in hist],
        })
    pooled = counts.sum(axis=0)
    return {
        "groups": groups,
        "tanh_saturation_ratio": int(saturated.sum()) / max(int(pooled.sum()), 1),
        "dim_entropy_bits": [entropy_bits(row) for row in pooled],
    }


class Measurements:
    def __init__(self, keys):
        self.keys = keys
        self.rows = []
        self.histograms = {}

    def cohorts(self, row):
        identity = tuple(row[k] for k in self.keys)
        yield (*identity, "all")
        if row["matched_six"]:
            yield (*identity, "matched_six")

    def add(self, row, counts=None, saturated=None):
        self.rows.append(row)
        if counts is None:
            return
        for identity in self.cohorts(row):
            total, sat = self.histograms.setdefault(identity, (np.zeros_like(counts), np.zeros_like(saturated)))
            total += counts
            sat += saturated

    def aggregate(self):
        by_key = defaultdict(list)
        for row in self.rows:
            for identity in self.cohorts(row):
                by_key[identity].append(row)
        summaries = []
        for identity, rows in by_key.items():
            summary = {**dict(zip((*self.keys, "cohort"), identity)), **summarize(rows)}
            if identity in self.histograms:
                summary["fsq"] = fsq_summary(*self.histograms[identity])
            summaries.append(summary)
        return summaries
