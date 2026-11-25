"""FIXED/CLOSED native quality and budget monotonicity."""

from pathlib import Path

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import MAX_K, ContinuousReceiver, RankOrder, encode_continuous, truncate
from scripts.hla_tok.continuous.inputs import METRICS, Inputs
from scripts.hla_tok.continuous.matrix import (
    Setting,
    compute,
    evaluate_matrix,
    groups,
    representative,
    summarize_matrix,
    validation_frames,
)
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.v3.gates import signed_gain


def _gain_frames(small: list[dict], large: list[dict], limits: dict) -> dict:
    return {
        metric: sum(value < -limits[metric] for value in deltas) / len(deltas)
        for metric, deltas in (
            ("global_psnr", [new["psnr"] - old["psnr"] for old, new in zip(small, large, strict=True)]),
            ("lpips", [old["lpips"] - new["lpips"] for old, new in zip(small, large, strict=True)]),
            ("dists", [old["dists"] - new["dists"] for old, new in zip(small, large, strict=True)]),
        )
    }


def _adjacent(rows: list[dict], limits: dict) -> list[dict]:
    by_budget = {key["budget"]: frames for key, frames in groups(rows, ("budget",))}
    pairs = list(zip(sorted(by_budget), sorted(by_budget)[1:]))
    result = []
    for lower, upper in pairs:
        small, large = by_budget[lower], by_budget[upper]
        gain = signed_gain(summarize(large), summarize(small))
        if [(r["video"], r["frame"]) for r in small] != [(r["video"], r["frame"]) for r in large]:
            raise ValueError("cannot compare unpaired quality rows")
        result.append({"lower": lower, "upper": upper, "gain": gain,
                       "pass": all(gain[key] >= -limits[key] for key in METRICS),
                       "frame_violation_fraction": _gain_frames(small, large, limits)})
    return result


def progressive(rows: list[dict], config: dict, baseline: dict) -> dict:
    limits = config["progressive"]["tolerances"]
    result = {}
    for key, frames in groups(rows, ("order",)):
        last, = [row for row in summarize_matrix(frames) if row["budget"] == MAX_K]
        adjacent = _adjacent(frames, limits)
        budgets = {row["budget"]: row for row in summarize_matrix(frames)}
        base, mid = config["budgets"][0], config["budgets"][2]
        gain = {"base_to_fixed_psnr": last["global_psnr"] - budgets[base]["global_psnr"],
                "base_to_fixed_lpips": budgets[base]["lpips"] - last["lpips"],
                "mid_to_fixed_psnr": last["global_psnr"] - budgets[mid]["global_psnr"]}
        checks = {
            "adjacent_average_monotonic": all(pair["pass"] for pair in adjacent),
            "substantial_gain": all(gain[k] >= v for k, v in config["progressive"]["minimum_gains"].items()),
            "not_saturated_in_top_three": any(pair["gain"][m] > limits[m] for pair in adjacent[-2:] for m in limits),
            "max_budget_matches": all(abs(last[m] - baseline[m]) <= config["reproduction"]["tolerances"][m]
                                      for m in METRICS),
        }
        result[key["order"]] = {
            "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "gain": gain, "max_budget_delta": signed_gain(last, baseline), "adjacent": adjacent,
            "by_source": {video["video"]: _adjacent(data, limits)
                          for video, data in groups(frames, ("video",))},
        }
    return result


@torch.inference_mode()
def verify_closed(inputs: Inputs, orders: list[RankOrder]) -> dict:
    # Pass each representative frame through separately-owned synthesis endpoints per budget.
    rows = []
    receivers = {budget: ContinuousReceiver.from_model(inputs.model) for budget in inputs.config["budgets"]}
    references = validation_frames(inputs, representative(inputs.cohort))
    with compute(inputs.config["quality_compute_dtype"]):
        shared_receiver = ContinuousReceiver.from_model(inputs.model)
        for video, index, source in references:
            full = encode_continuous(inputs.model, source)
            for order in orders:
                shared = order.apply(full)
                independent = order.apply(encode_continuous(inputs.model, source))
                for count, receiver in receivers.items():
                    fixed = shared_receiver.decode(truncate(shared, count), order)
                    closed = receiver.decode(truncate(independent, count), order)
                    rows.append({"video": video, "frame": index, "order": order.name.name, "budget": count,
                                 "rgb_bit_exact": torch.equal(fixed, closed),
                                 "nonfinite": not bool(torch.isfinite(closed).all())})
    if any(not row["rgb_bit_exact"] or row["nonfinite"] for row in rows):
        raise ValueError("independent C0 decode differs from stateless shared decode")
    return {
        "status": "PASS", "temporal_state": "NOT_APPLICABLE", "comparisons": rows,
        "sender_state": None, "receiver_state": None, "state_norm": None, "state_drift": "NOT_APPLICABLE",
        "earlier_budget_affects_encoding": False, "requires_feedback": False,
        "scope": "FIXED and CLOSED collapse to the same stateless full-cohort evaluation",
    }


def _clip_periods(frames: list[dict]) -> dict:
    spans = {"full": frames, "early": frames[:len(frames) // 3],
             "middle": frames[len(frames) // 3:2 * len(frames) // 3], "late": frames[2 * len(frames) // 3:]}
    stats = {name: summarize(data) for name, data in spans.items()}
    return stats | {"late_minus_early": signed_gain(stats["late"], stats["early"]),
                    "closed_minus_fixed": dict.fromkeys(METRICS, 0.0)}


def run_budgets(inputs: Inputs, quality: NativeQuality, output: Path, orders: list[RankOrder], name: str) -> list[dict]:
    settings = [Setting(order, budget) for order in orders for budget in inputs.config["budgets"]]
    samples = output / "samples" if name == "native" else None
    rows = evaluate_matrix(inputs, quality, settings, output, name, samples=samples)
    prefix = "" if name == "native" else "ordered_"
    for mode in ("fixed", "closed"):
        labelled = [row | {"mode": mode.upper()} for row in rows]
        report = [row | {"mode": mode.upper()} for row in summarize_matrix(labelled)]
        write_csv(output / f"{prefix}{mode}_budget.csv", report)
        write_csv(output / f"{prefix}{mode}_budget_per_frame.csv", labelled)
        write_csv(output / f"{prefix}{mode}_budget_by_video.csv", summarize_matrix(labelled, by_video=True))
    audit = verify_closed(inputs, orders)
    clips = [key | _clip_periods(frames) for key, frames in groups(rows, ("order", "budget", "video"))]
    write_json(output / f"{prefix}closed_state_audit.json", inputs.provenance() | audit | {"clips": clips})
    return rows
