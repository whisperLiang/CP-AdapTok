"""Budget switches on the three real contiguous native Validation clips."""

from pathlib import Path
from random import Random

import torch
from torch import Tensor

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import ContinuousReceiver, RankOrder, encode_continuous, stream_frames, truncate
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute, groups, validation_frames
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.continuous.quality import summarize
from scripts.hla_tok.v2.protocol import csv_at, json_at


def plans(budgets: list[int], count: int, config: dict, video_number: int) -> dict[str, list[int]]:
    low, high = budgets[0], budgets[-1]
    ranks = [min(len(budgets) - 1, index * len(budgets) // count) for index in range(count)]
    random = Random(config["seed"] + video_number)
    switch = 2 * count // 3
    return {
        "constant": [config["constant_budget"]] * count,
        "descending": [budgets[-rank - 1] for rank in ranks],
        "ascending": [budgets[rank] for rank in ranks],
        "alternate": ([high, low] * count)[:count],
        "random": random.choices(budgets, k=count),
        "recover": [low] * switch + [high] * (count - switch),
    }


def _fixed_lookup(output: Path) -> dict:
    data = csv_at(output / "fixed_budget_per_frame.csv") + csv_at(output / "ordered_fixed_budget_per_frame.csv")
    return {(row["order"], row["video"], int(row["frame"]), int(row["budget"])):
            {name: float(row[name]) for name in ("mse", "psnr", "lpips", "dists")} for row in data}


def _independent(model, receiver: ContinuousReceiver, source: Tensor, order: RankOrder,
                 budgets: set[int]) -> dict:
    values = order.apply(encode_continuous(model, source))
    return {budget: (prefix, receiver.decode(prefix, order)) for budget in budgets
            for prefix in (truncate(values, budget),)}


def _check_output(stream_result: tuple[Tensor, Tensor], expected: tuple[Tensor, Tensor]) -> None:
    if not all(torch.equal(left, right) for left, right in zip(stream_result, expected, strict=True)):
        raise ValueError("budget switching changed the stateless frame result")
    if not bool(torch.isfinite(stream_result[1]).all()):
        raise ValueError("nonfinite dynamic reconstruction")


def _evaluate_clip(inputs: Inputs, order: RankOrder, video: str, frames: list[Tensor],
                   schedules: dict[str, list[int]], fixed: dict) -> list[dict]:
    receiver = ContinuousReceiver.from_model(inputs.model)
    streams = {name: stream_frames(inputs.model, zip(frames, budgets, strict=True), order)
               for name, budgets in schedules.items()}
    records = []
    for index, source in enumerate(frames):
        current_budgets = {name: budgets[index] for name, budgets in schedules.items()}
        reference = _independent(inputs.model, receiver, source, order, set(current_budgets.values()))
        for name, budget in current_budgets.items():
            _check_output(next(streams[name]), reference[budget])
            metrics = fixed[(order.name.name, video, index, budget)]
            records.append({"order": order.name.name, "video": video, "frame": index, "schedule": name,
                            "budget": budget, "rgb_and_tokens_equal_fixed": True, "nonfinite": False} | metrics)
    return records


@torch.inference_mode()
def dynamic_phase(inputs: Inputs, output: Path) -> None:
    names = json_at(output / "ordering_decision.json")["wire_orders"]
    orders = [order for order in load_orders(output) if order.name.name in names]
    fixed = _fixed_lookup(output)
    records = []
    with compute(inputs.config["quality_compute_dtype"]):
        for number, (video, length) in enumerate(inputs.cohort.items()):
            sources = validation_frames(inputs, ((video, index) for index in range(length)))
            frames = [frame for _, _, frame in sources]
            schedules = plans(inputs.config["budgets"], length, inputs.config["dynamics"], number)
            for order in orders:
                records.extend(_evaluate_clip(inputs, order, video, frames, schedules, fixed))
            print(f"Dynamic {video}: {length} frames, {len(schedules)} schedules, {len(orders)} orders", flush=True)
    write_csv(output / "dynamic_budget_per_frame.csv", records)
    scopes = ("order", "schedule")
    write_csv(output / "dynamic_budget.csv", [key | summarize(rows) for key, rows in groups(records, scopes)])
    write_csv(output / "dynamic_budget_by_video.csv",
              [key | summarize(rows) for key, rows in groups(records, (*scopes, "video"))])
    write_json(output / "dynamic_state_audit.json", inputs.provenance() | {
        "status": "PASS", "schedules": sorted({row["schedule"] for row in records}), "orders": names,
        "separate_clip_boundaries": list(inputs.cohort), "checked_outputs": len(records),
        "all_tokens_and_rgb_equal_to_independent_stateless_decode": True,
        "quality_reused_from_identical_source_order_and_budget": True,
        "persistent_state": "NOT_APPLICABLE", "requires_ack": False, "nonfinite_outputs": 0,
        "same_frame_retransmission_required": False, "recovery_is_immediate_on_the_next_high_budget_frame": True,
    })
