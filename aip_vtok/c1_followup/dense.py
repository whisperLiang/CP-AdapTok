"""Rank truncation, same-history replay, and dense diagnostic measurements."""

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from aip_vtok.c1.channels import aligned_value, coordinate_value
from aip_vtok.c1.frames import capture, frame_fields
from aip_vtok.c1.metrics import FrameMetrics
from aip_vtok.c1.recovery_net import commit_reference
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.diagnostics.diag0.metrics import summarize
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.training.losses import bt709_chroma

from .quantization import CARDINALITY, Quantizer, Stage, dense_ledger

SHARED = "same_context_b0_fsq_history"
ARMS = ("A0", "A1", "A2", "A3", "A4-cont", "A4-fsq", "A5-cont", "A5-fsq")
BASELINE_NAMES = {"B0-cont": "A0", "B0-fsq": "A1"}
MAX_REPLAY_MSE_ERROR = 5e-9


@dataclass(frozen=True)
class Layout:
    indices: torch.Tensor
    channels: int
    space: str
    quantizer: Quantizer | None = None

    @property
    def groups(self):
        return len(self.indices) // 6

    def select(self, sample):
        source = coordinate_value(sample.native, sample.aligned, self.space)
        return source.index_select(1, self.indices).unflatten(1, (-1, 6)).movedim(2, -1)

    def encode(self, sample):
        values = self.select(sample)
        return ((values,), ()) if self.quantizer is None else self.quantizer.encode(values)

    def decode_values(self, payload):
        return payload[0] if self.quantizer is None else self.quantizer.decode(*payload)

    def restore(self, values, backbone):
        ordered = values.movedim(-1, 2).flatten(1, 2)
        base = ordered.new_zeros(self.channels).view(1, -1, 1, 1)
        base = base.expand(len(ordered), -1, *ordered.shape[-2:])
        return aligned_value(backbone, base.index_copy(1, self.indices, ordered), self.space)

    def decode(self, payload, backbone):
        return self.restore(self.decode_values(payload), backbone)

    def record(self):
        return {
            "indices": self.indices.tolist(), "channels": self.channels, "space": self.space,
            "stages": [] if self.quantizer is None else [stage.record() for stage in self.quantizer.stages],
        }

    @classmethod
    def load(cls, record, device):
        stages = [Stage.load(stage, device) for stage in record["stages"]]
        quantizer = Quantizer(*stages) if stages else None
        return cls(torch.tensor(record["indices"], device=device), record["channels"], record["space"], quantizer)


def timed(call, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call(*args)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1_000


@torch.no_grad()
def shared_frames(codec, clip, baseline):
    state = ReferenceState()
    backbone = codec.backbone
    for index, image in enumerate(clip):
        sample, capture_ms = timed(capture, backbone, state, index, image)
        layout = baseline[sample.frame_type]
        yield sample, capture_ms
        payload, _ = layout.encode(sample)
        reference = layout.decode(payload, backbone)
        state = commit_reference(sample.context, reference, codec.projector(reference, sample.context.intra))


def chroma_planes(rgb, rec):
    source, target = bt709_chroma(rgb), bt709_chroma(rec)
    dims = (0, 2, 3)
    variance = 100 * target.var(dims, correction=0) / source.var(dims, correction=0).clamp_min(1e-10)
    mse = (source - target).square().mean(dims)
    bias = (source.mean(dims) - target.mean(dims)).abs()
    return {f"{plane}_{field}": value
            for field, pair in zip(("var_retention_pct", "mse", "mean_abs_bias"), (variance, mse, bias))
            for plane, value in zip(("cb", "cr"), pair.tolist())}


def decode_frame(backbone, layout, payload, context):
    latent = layout.decode(payload, backbone)
    rgb, _ = backbone.decode_aligned(latent, context)
    return rgb, latent


def symbol_histogram(stage, ids):
    groups = len(stage.levels)
    offset = torch.arange(groups, device=ids.device).view(1, -1, 1, 1)
    counts = torch.bincount((ids + offset * CARDINALITY).flatten(), minlength=groups * CARDINALITY)
    return counts.reshape(groups, CARDINALITY)


def marginal_entropy(counts):
    probabilities = counts.double() / counts.sum(-1, keepdim=True).clamp_min(1)
    return -(probabilities * probabilities.clamp_min(1e-30).log2()).sum(-1)


class Histograms:
    def __init__(self):
        self.totals = {}

    def add_stage(self, key, stage, ids, saturated):
        groups = len(stage.levels)
        bins = max(map(max, stage.levels))
        offsets = torch.arange(groups * 6, device=ids.device).view(1, groups, 1, 1, 6) * bins
        digits = stage.fsq.ids_to_digits(ids)
        histogram = torch.bincount((digits + offsets).flatten(), minlength=groups * 6 * bins)
        dims = histogram.reshape(groups, 6, bins)
        codes = symbol_histogram(stage, ids)
        sat = saturated.sum((0, 2, 3))
        if key not in self.totals:
            self.totals[key] = (stage.levels, dims.clone(), codes.clone(), sat.clone())
        else:
            for total, value in zip(self.totals[key][1:], (dims, codes, sat)):
                total.add_(value)
        return marginal_entropy(codes)

    def record(self):
        return [{"arm": arm, "frame_type": frame_type, "level": level,
                 "groups": group_histograms(*data)}
                for (arm, frame_type, level), data in sorted(self.totals.items())]


def group_histograms(levels, dims, codes, sat):
    group_stats = zip(levels, dims.tolist(), marginal_entropy(codes).tolist(), sat.tolist())
    return [
        {"group": group, "id_entropy_bits": entropy,
         "dimensions": [dimension_histogram(hist[:bins], count) for hist, bins, count in zip(rows, lev, sats)]}
        for group, (lev, rows, entropy, sats) in enumerate(group_stats)
    ]


def dimension_histogram(hist, saturation):
    total = sum(hist)
    return {
        "counts": hist, "entropy_bits": -sum(n / total * math.log2(n / total) for n in hist if n),
        "boundary_code_ratio": (hist[0] + hist[-1]) / total,
        "tanh_saturation_ratio": saturation / total,
    }


def add_rate(histograms, layout, payload, saturated, arm, kind, shape):
    stages = () if layout.quantizer is None else layout.quantizer.stages
    entropy = [
        histograms.add_stage((arm, kind, index), stage, ids, sat)
        for index, (stage, ids, sat) in enumerate(zip(stages, payload, saturated), 1)
    ]
    avg = torch.stack(entropy).mean().item() if entropy else None
    return {
        **dense_ledger(layout.groups, len(stages), shape, avg),
        "entropy_payload_bits_per_position": None if avg is None else avg * layout.groups * len(stages),
    }


def arm_row(backbone, sample, layout, fields, scorer, histograms, capture_ms):
    (payload, sat), symbol_ms = timed(layout.encode, sample)
    (rgb, latent), decode_ms = timed(decode_frame, backbone, layout, payload, sample.context)
    return {
        **fields, **scorer(rgb, latent, sample.aligned), **chroma_planes(sample.rgb, rgb),
        **add_rate(histograms, layout, payload, sat, fields["arm"], sample.frame_type, sample.aligned.shape[-2:]),
        "encode_ms": capture_ms + symbol_ms, "backbone_encode_ms": capture_ms,
        "symbol_encode_ms": symbol_ms, "decode_ms": decode_ms,
    }


@torch.no_grad()
def evaluate_samples(codec, samples, arms, suite, histograms):
    for video, split, sample, capture_ms in samples:
        scorer = FrameMetrics(suite, sample.rgb)
        for name, layouts in arms.items():
            fields = {**frame_fields(sample, video, SHARED, split), "arm": name}
            yield arm_row(codec.backbone, sample, layouts[sample.frame_type], fields, scorer, histograms, capture_ms)


def quality_summary(rows):
    scopes = {"all": rows, "I": [row for row in rows if row["frame_type"] == "I"],
              "P": [row for row in rows if row["frame_type"] == "P"]}
    return {name: summarize(group) for name, group in scopes.items() if group}


def summarize_arms(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    return {arm: quality_summary(group) for arm, group in grouped.items()}


def frame_key(row):
    return row["video"], int(row["frame"]), row["frame_type"]


def replay_comparison(actual, expected):
    actual_by_frame = {frame_key(row): row["mse"] for row in actual}
    prior_by_frame = {frame_key(row): row["mse"] for row in expected}
    same_ids = actual_by_frame.keys() == prior_by_frame.keys() and len(actual_by_frame) == len(actual) == len(expected)
    max_error = max((abs(mse - prior_by_frame[key]) for key, mse in actual_by_frame.items()), default=0) if same_ids else None
    return {
        "same_frame_ids": same_ids, "max_per_frame_mse_error": max_error,
        "measured": distortion_summary(actual) if actual else None,
        "sealed": distortion_summary(expected) if expected else None,
        "passed": bool(actual) and same_ids and max_error <= MAX_REPLAY_MSE_ERROR,
    }


def historical_rows(sealed_path):
    return [row for row in json_at(sealed_path)["rows"]
            if row["history"] == SHARED and row["method"] in BASELINE_NAMES and row["seed"] == 0]


def for_arm(rows, field, value):
    return [row for row in rows if row[field] == value]


def check_replay(rows, sealed_path):
    old = historical_rows(sealed_path)
    comparisons = {
        arm: replay_comparison(for_arm(rows, "arm", arm), for_arm(old, "method", name))
        for name, arm in BASELINE_NAMES.items()
    }
    return {"passed": all(result["passed"] for result in comparisons.values()),
            "mse_absolute_tolerance": MAX_REPLAY_MSE_ERROR, "comparisons": comparisons}


@torch.no_grad()
def reconstruction_mse(backbone, sample, layout):
    payload, _ = layout.encode(sample)
    rgb, _ = decode_frame(backbone, layout, payload, sample.context)
    return F.mse_loss(rgb, sample.rgb).item()
