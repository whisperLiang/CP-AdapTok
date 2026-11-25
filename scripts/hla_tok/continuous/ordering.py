"""Static ordering from Train-only leave-one-Rank-out reconstruction."""

import json
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import mean

import torch
from torch import Tensor
from torch.nn import functional as F

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.models.codec import recover_grid
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.continuous import (
    MAX_K,
    NATIVE,
    NATIVE_RANKS,
    REGIONS,
    ContinuousReceiver,
    Order,
    RankOrder,
    encode_continuous,
)
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3.gates import signed_gain


def _train_plan(inputs: Inputs) -> list[dict]:
    config = inputs.config["calibration"]
    catalog = inputs.reader.catalog("train")
    eligible = sorted(video for video, row in catalog.items()
                      if tuple(reversed(row["resolution"])) == NATIVE_SHAPE and not row["resized"])
    rng = Random(config["seed"])
    samples = []
    for video in rng.sample(eligible, config["sources"]):
        records = catalog[video]["frame_hashes"]
        index = rng.randrange(len(records))
        samples.append({"video": video, "frame": index, "source_sha256": records[index]["sha256"]})
    return samples


def _remove(tokens: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    start = rank * REGIONS
    selected = torch.cat((tokens[:, :start], tokens[:, start + REGIONS:]), dim=1)
    ranks = torch.tensor([r for r in NATIVE_RANKS if r != rank], device=tokens.device)
    return selected, ranks


def _rank_distortions(inputs: Inputs, receiver: ContinuousReceiver, photo: Tensor, native: RankOrder) -> list[float]:
    tokens = encode_continuous(inputs.model, photo)
    full_mse = F.mse_loss(receiver.decode(tokens, native), photo).item()
    differences = []
    for rank in NATIVE_RANKS:
        kept, ranks = _remove(tokens, rank)
        features = recover_grid(receiver.local_decoder, kept, NATIVE, ranks)
        error = F.mse_loss(receiver.reconstruct(features), photo).item()
        differences.append(error - full_mse)
    return differences


def _permutation_digest(ranks) -> str:
    return sha256(json.dumps(ranks, separators=(",", ":")).encode()).hexdigest()


def _order_record(order: RankOrder) -> dict:
    inverse = [order.ranks.index(rank) for rank in NATIVE_RANKS]
    return {"order": order.name.name, "wire_to_native_rank": order.ranks, "native_to_wire_rank": inverse,
            "sha256": _permutation_digest(order.ranks)}


@torch.inference_mode()
def calibrate(inputs: Inputs, output: Path) -> list[RankOrder]:
    plan = _train_plan(inputs)
    receiver = ContinuousReceiver.from_model(inputs.model)
    native = RankOrder.native("cuda")
    scores = []
    before = dict(inputs.reader.counts)
    with compute(inputs.config["quality_compute_dtype"]):
        for sample in plan:
            source = transfer(inputs.reader.read("train", sample["video"], sample["frame"])[None])
            scores.append(_rank_distortions(inputs, receiver, source, native))
            print(f"Train calibration {len(scores)}/{len(plan)}", flush=True)
    reads = {key: inputs.reader.counts[key] - before[key] for key in before}
    if reads != {"train": len(plan), "validation": 0, "test": 0}:
        raise ValueError("static ordering must be calibrated only on admitted Train frames")
    importance = [mean(values) for values in zip(*scores, strict=True)]
    ranked = tuple(sorted(NATIVE_RANKS, key=lambda rank: (-importance[rank], rank)))
    orders = [native, RankOrder.reversed("cuda"), RankOrder(Order.O2, ranked, "cuda")]
    write_csv(output / "rank_importance_train.csv", [
        {"original_rank": rank, "actual_regions": min(REGIONS, MAX_K - rank * REGIONS),
         "mean_delta_rgb_mse": importance[rank], "train_frames": len(plan),
         **{f"train_{i:02d}_delta": row[rank] for i, row in enumerate(scores)}} for rank in NATIVE_RANKS
    ])
    write_json(output / "rank_order_manifest.json", inputs.provenance() | {
        "calibration_split": "train", "samples": plan, "pixel_reads": reads,
        "compute_dtype": inputs.config["quality_compute_dtype"], "score": inputs.config["calibration"]["score"],
        "ordering": [_order_record(order) for order in orders],
    })
    write_csv(output / "rank_ordering.csv", [
        {"order": order.name.name, "wire_rank": position, "native_rank": rank,
         "actual_regions": min(REGIONS, MAX_K - rank * REGIONS)}
        for order in orders for position, rank in enumerate(order.ranks)
    ])
    return orders


def load_orders(folder: Path, device: str = "cuda") -> list[RankOrder]:
    orders = []
    for entry in json_at(folder / "rank_order_manifest.json")["ordering"]:
        ranks = tuple(entry["wire_to_native_rank"])
        if _permutation_digest(ranks) != entry["sha256"]:
            raise ValueError("frozen Rank order SHA-256 mismatch")
        order = RankOrder(Order[entry["order"]], ranks, device)
        if _order_record(order)["native_to_wire_rank"] != entry["native_to_wire_rank"]:
            raise ValueError("frozen Rank order inverse mismatch")
        orders.append(order)
    return orders


def _read_qualities(path: Path) -> dict:
    return {(row["order"], int(row["budget"])): {m: float(row[m]) for m in ("global_psnr", "lpips", "dists")}
            for row in csv_at(path)}


def decide_orders(inputs: Inputs, output: Path, orders: list[RankOrder], progressive_result: dict) -> dict:
    limits = inputs.config["progressive"]
    rows = _read_qualities(output / "fixed_budget.csv") | _read_qualities(output / "ordered_fixed_budget.csv")
    budgets = limits["low_budget_comparison"]
    gains = []
    eligible = ["O0"]
    for order in orders[1:]:
        name = order.name.name
        for budget in budgets:
            delta = signed_gain(rows[(name, budget)], rows[("O0", budget)])
            material = (any(delta[m] >= limits["low_budget_ordering_gain"][m] for m in delta)
                        and all(delta[m] >= -limits["tolerances"][m] for m in delta))
            gains.append({"order": name, "budget": budget, "material_improvement": material} | delta)
        if (progressive_result[name]["checks"]["max_budget_matches"]
                and any(row["material_improvement"] for row in gains if row["order"] == name)):
            eligible.append(name)
    primary = max(eligible, key=lambda name: mean(rows[(name, budget)]["global_psnr"] for budget in budgets))
    decision = {"wire_orders": eligible, "primary_order": primary, "signed_low_budget_gains": gains}
    write_json(output / "ordering_decision.json", decision)
    return decision
