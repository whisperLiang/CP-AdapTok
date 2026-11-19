"""Score fixed and independently committed histories on the same frame inventory."""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import torch

from aip_vtok.c1.artifacts import timestamp
from aip_vtok.c1.dense_eval import validation_videos
from aip_vtok.c1.metrics import FrameMetrics, quality_backslides
from aip_vtok.c1.scan import BUDGETS, REF_BUDGET
from aip_vtok.c1_followup.dense import chroma_planes, quality_summary
from aip_vtok.c1_followup.io import write_rows
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rms
from aip_vtok.runtime import file_hash, write_json

from .calibration import has_progress, load_scan
from .protocol import COHORT_FRAMES, HORIZON, LIMITS, OUTPUT, S1_GATE, SCAN_LOCK
from .receiver import B2, B3, ZERO, Receiver
from .rollout import CausalPair, LinkFrame
from .scan import LEGACY, POLICIES

FIXED, CLOSED = "fixed_reference", "independent_closed"
S1 = OUTPUT / "s1"
S2 = OUTPUT / "s2"
KEYS = ("strategy", "method", "variant", "history")


@dataclass(frozen=True)
class Trial:
    strategy: str
    method: str
    cases: dict
    receiver: Receiver
    variant: str = "base"

    def history(self, history):
        return f"{history}_B3" if self.method == B3 else history

    def fields(self, history):
        return {"strategy": self.strategy, "method": self.method, "variant": self.variant,
                "history": self.history(history)}


def reference_stats(pair, frame, trial):
    budget = REF_BUDGET[frame.context.intra]
    rate = trial.cases[frame.sample.frame_type].rates[budget]
    fields = {"reference_budget": budget, "state_sync": True,
              "reference_strategy": trial.strategy, "reference_method": trial.method,
              "reference_payload_bits": rate["payload_bits"],
              "reference_entropy_payload_bits": rate["entropy_payload_bits"],
              "reference_latent_rms": rms(pair.rx.latent), "projected_feature_rms": rms(pair.rx.feature),
              "prefix_ids_match": True, "mask_dependency": True}
    if frame.context.frame_id in (31, 63):
        fields.update(tx_reference_sha256=pair.tx.fingerprint(), rx_reference_sha256=pair.rx.fingerprint())
    return fields


def check_finite(values):
    if any(not math.isfinite(value) for value in values if isinstance(value, float)):
        raise RuntimeError("nonfinite sparse reconstruction, rate or reference metric")


class FrameScorer:
    """Reuse reconstruction metrics only for bitwise-equal decoder inputs on one frame."""

    def __init__(self, codec, suite, rgb):
        self.codec = codec
        self.metrics = FrameMetrics(suite, rgb)
        self.contexts = []

    def context_cache(self, frame):
        ctx, target = frame.context.context, frame.sample.aligned
        for previous, previous_target, cache in self.contexts:
            same_context = ctx is previous or (ctx is not None and previous is not None and torch.equal(ctx, previous))
            if same_context and torch.equal(target, previous_target):
                return cache
        cache = defaultdict(list)
        self.contexts.append((ctx, target, cache))
        return cache

    def measure(self, frame, latent, cache):
        for previous, score in cache:
            if torch.equal(previous, latent):
                return score
        rec, _ = self.codec.backbone.decode_aligned(latent, frame.context)
        sample = frame.sample
        score = self.metrics(rec, latent, sample.aligned) | chroma_planes(sample.rgb, rec)
        cache.append((latent, score))
        return score


def score_frame(codec, frame, reference, scorer, video, trial, history):
    sample = frame.sample
    case = trial.cases[sample.frame_type]
    common = {"video": video, "frame": frame.context.frame_id, "frame_type": sample.frame_type,
              "split": "validation", **trial.fields(history), **reference}
    cache = scorer.context_cache(frame)
    for k in BUDGETS:
        latent = trial.receiver(case, codec.backbone, frame.context, frame.payload[:, :k])
        wire = {}
        if history == CLOSED or trial.strategy == LEGACY:
            rate = case.rates[max(k, reference["reference_budget"])]
            wire = {"maintaining_reference_payload_bits": rate["payload_bits"],
                    "maintaining_reference_entropy_payload_bits": rate["entropy_payload_bits"]}
        row = {**common, "budget": k, **case.rates[k],
               **wire, **scorer.measure(frame, latent, cache[k])}
        check_finite(row.values())
        yield row


@torch.no_grad()
def video_rows(codec, suite, clip, video, legacy, trials):
    shared = CausalPair()
    pairs = {index: CausalPair() for index, trial in enumerate(trials) if trial.strategy != LEGACY}
    for index, image in enumerate(clip):
        frame = shared.advance(codec, image, index, legacy.cases, legacy.receiver)
        anchor = reference_stats(shared, frame, legacy)
        scorer = FrameScorer(codec, suite, frame.sample.rgb)
        packets = {LEGACY: frame.payload}
        for trial_id, trial in enumerate(trials):
            if trial.strategy not in packets:
                packets[trial.strategy] = trial.cases[frame.sample.frame_type].encode(frame.sample)
            fixed = LinkFrame(frame.sample, frame.context, packets[trial.strategy])
            rows = list(score_frame(codec, fixed, anchor, scorer, video, trial, FIXED))
            yield from rows
            if trial_id not in pairs:
                # The frozen A1/B2 control is also the source of the fixed history.
                yield from (row | trial.fields(CLOSED) for row in rows)
                continue
            pair = pairs[trial_id]
            closed = pair.advance(codec, image, index, trial.cases, trial.receiver)
            stats = reference_stats(pair, closed, trial)
            yield from score_frame(codec, closed, stats, scorer, video, trial, CLOSED)


def validate_inventory(rows, counts, trials):
    scopes = {(t.strategy, t.method, t.variant, t.history(mode)) for t, mode in product(trials, (FIXED, CLOSED))}
    frames = [(video, frame) for video, count in counts.items() for frame in range(count)]
    expected = {(*scope, *frame, k) for scope, frame, k in product(scopes, frames, BUDGETS)}
    actual = {tuple(row[k] for k in (*KEYS, "video", "frame", "budget")) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("requires exactly one row per trial, history, frame and budget")
    if not health_passed(rows):
        raise RuntimeError("sparse reference/prefix/mask integrity failed")


def health_passed(rows):
    return all(all(row[k] for k in ("state_sync", "prefix_ids_match", "mask_dependency")) for row in rows)


def evaluate(run, lock, trials):
    if lock != load_scan():
        raise RuntimeError("Validation requires the frozen, Train-only scan")
    run.reader.validation_locked = True
    catalog = run.reader.open("validation")
    counts = validation_videos(run.config, catalog)
    if counts.get(HORIZON_VIDEO) != HORIZON or sum(counts.values()) != COHORT_FRAMES or run.config["gop"] != 32:
        raise RuntimeError("expected 174 Validation frames including the complete 64-frame GOP32 horizon")
    rows = []
    for video, count in counts.items():
        clip = run.reader.read(catalog, video, 0, count)
        rows.extend(video_rows(run.codec, run.suite, clip, video, legacy_trial(run), trials))
        print(f"Validation {video}: {count} frames, {len(trials)} methods, two histories", flush=True)
    validate_inventory(rows, counts, trials)
    return rows, counts


def summarize_points(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in (*KEYS, "budget"))].append(row)
    return [{**dict(zip((*KEYS, "budget"), identity)), **quality_summary(chunk)} for identity, chunk in groups.items()]


def efficiency(points):
    groups = defaultdict(list)
    for p in points:
        groups[tuple(p[k] for k in KEYS)].append(p)
    for identity, group in groups.items():
        ordered = sorted(group, key=lambda p: p["budget"])
        for lo, hi in zip(ordered, ordered[1:]):
            lower, upper = lo["all"], hi["all"]
            mse = lower["mse"] - upper["mse"]
            fixed = upper["payload_bits"] - lower["payload_bits"]
            entropy = upper["entropy_payload_bits"] - lower["entropy_payload_bits"]
            yield {**dict(zip(KEYS, identity)), "lower": lo["budget"], "upper": hi["budget"],
                   "mse_drop": mse, "fixed_additional_bits": fixed, "entropy_additional_bits": entropy,
                   "delta_mse_per_fixed_bit": mse / fixed, "delta_mse_per_entropy_bit": mse / entropy}


def summaries(rows):
    horizon = [row for row in rows if row["video"] == HORIZON_VIDEO]
    result = {}
    for name, chunk in (("validation", rows), ("horizon", horizon)):
        points = summarize_points(chunk)
        result[name] = {"points": points, "efficiency": list(efficiency(points)),
                        "backslides": quality_backslides(chunk, KEYS)}
    result["late"] = summarize_points([row for row in horizon if 48 <= row["frame"] < HORIZON])
    result["reference_frames"] = [r for r in horizon if r["frame"] in (31, 63) and r["budget"] == BUDGETS[0]]
    return result


def legacy_trial(run):
    return Trial(LEGACY, B2, run.inputs.legacy, Receiver(run.inputs.legacy_sigma))


def s1_trials(run, lock):
    trials = []
    for name in POLICIES:
        cases = run.inputs.cases(name, lock["keys"][name])
        trials.extend((Trial(name, B2, cases, Receiver(lock["sigma"])), Trial(name, ZERO, cases, Receiver(None))))
    return [*trials, legacy_trial(run)]


def balanced_curve(points, winner, history):
    group = sorted((r for r in points if (r["strategy"], r["method"], r["history"]) == (winner, B2, history)),
                   key=lambda r: r["budget"])
    return [(r["I"]["psnr_global_mse"] + r["P"]["psnr_global_mse"]) / 2 for r in group]


def s1_decision(summary, rows, winner):
    if winner is None:
        return {"released": False, "winner": None, "reason": "no progressive B2 candidate on Train"}
    curves = {history: balanced_curve(summary["validation"]["points"], winner, history)
              for history in (FIXED, CLOSED)}
    valid = all(has_progress(curve) for curve in curves.values())
    max_rms = max(r["reference_latent_rms"] for r in rows
                  if (r["strategy"], r["method"], r["history"]) == (winner, B2, CLOSED))
    released = valid and max_rms <= LIMITS["max_reference_latent_rms"]
    return {"released": released, "winner": winner, "balanced_curves_db": curves, "max_reference_latent_rms": max_rms,
            "reason": None if released else "Train winner failed Validation progression or reference RMS"}


def run_s1(run, lock):
    if S1_GATE.is_file():
        return load_s1(lock)
    rows, counts = evaluate(run, lock, s1_trials(run, lock))
    summary = summaries(rows)
    write_rows(S1, rows, summaries=summary, cohort=counts, scan_lock_sha256=file_hash(SCAN_LOCK))
    decision = s1_decision(summary, rows, lock["winner"])
    gate = {**decision, "s1_sha256": file_hash(S1.with_suffix(".json")), "scan_lock_sha256": file_hash(SCAN_LOCK),
            "integrity": run.verify_integrity(), "finished": timestamp()}
    write_json(S1_GATE, gate)
    print(f"S1 release: {decision}", flush=True)
    return gate


def load_s1(lock):
    gate = json_at(S1_GATE)
    source = S1.with_suffix(".json")
    if (gate["scan_lock_sha256"] != file_hash(SCAN_LOCK) or gate["s1_sha256"] != file_hash(source)
            or gate["winner"] != lock["winner"] or not gate["integrity"]["passed"]):
        raise RuntimeError("S1 gate is incomplete or disagrees with its frozen inputs")
    data = json_at(source)
    decision = s1_decision(data["summaries"], data["rows"], lock["winner"])
    if any(gate[key] != value for key, value in decision.items()):
        raise RuntimeError("S1 release disagrees with its Validation evidence")
    return gate
