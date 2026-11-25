"""Frozen continuous evaluation sequence, with the raw V3 replay as its gate."""

import subprocess
import sys
from pathlib import Path

import torch

from hla_tok.evaluation.io import write_json
from hla_tok.streaming.continuous import MAX_K, RankOrder
from scripts.hla_tok.continuous.budgets import progressive, run_budgets
from scripts.hla_tok.continuous.contracts import audit_causality, audit_prefix
from scripts.hla_tok.continuous.inputs import Inputs, open_inputs, record_replay
from scripts.hla_tok.continuous.ordering import calibrate, decide_orders, load_orders
from scripts.hla_tok.continuous.quality import NativeQuality
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at


def replay(v3: Path, output: Path) -> None:
    from scripts.hla_tok.continuous.inputs import CONFIG, CONFIG_SHA256, require_digest
    from scripts.hla_tok.v3.protocol import read_config

    require_digest(CONFIG, CONFIG_SHA256)
    checkpoint = v3 / "checkpoints/hla_tok/v3/quantized/seed1/C0/step_10000.pt"
    require_digest(checkpoint, read_config(CONFIG)["checkpoint_sha256"]["C0"])
    subprocess.run([sys.executable, "-m", "scripts.hla_tok.v3.run", "evaluate",
                    "--checkpoint", str(checkpoint), "--output", str(output / "native_replay")],
                   cwd=ROOT, check=True)
    record_replay(v3, output)


def _index(rows, budget):
    return {(row["video"], int(row["frame"])): row for row in rows if int(row["budget"]) == budget}


def _score_parity(output: Path, rows: list[dict]) -> None:
    reference = _index(csv_at(output / "native_replay/per_frame.csv"), MAX_K)
    measured = _index(rows, MAX_K)
    if reference.keys() != measured.keys():
        raise ValueError("maximum-budget quality was not measured on the frozen Validation pixels")
    fields = ("mse", "psnr", "lpips", "dists")
    differences = {field: max(abs(float(old[field]) - float(measured[key][field]))
                              for key, old in reference.items()) for field in fields}
    if any(differences.values()):
        raise ValueError(f"new maximum-budget scoring differs from V3: {differences}")
    write_json(output / "metrics_parity.json", {"status": "PASS", "frames": len(reference),
                                               "maximum_per_frame_absolute_delta": differences})


def native_phase(v3: Path, output: Path) -> None:
    if json_at(output / "reproduction.json")["status"] != "PASS":
        raise RuntimeError("Phase 0 must pass before multi-budget evaluation")
    inputs = open_inputs(v3)
    rows = run_budgets(inputs, NativeQuality(), output, [RankOrder.native("cuda")], "native")
    _score_parity(output, rows)
    inputs.finish(output / "native_run.json")


def _native_results(inputs: Inputs, output: Path) -> list[dict]:
    report = json_at(output / "native_run.json")
    if any(report[key] != value for key, value in inputs.provenance().items()):
        raise ValueError("Phase 1 native evidence has a different identity")
    rows = [row | {"budget": int(row["budget"]), "frame": int(row["frame"]),
                   **{m: float(row[m]) for m in ("mse", "psnr", "lpips", "dists")}}
            for row in csv_at(output / "fixed_budget_per_frame.csv")]
    _score_parity(output, rows)
    return rows


def _run_deterministically(callback, *args, **kwargs) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        callback(*args, **kwargs)
    finally:
        torch.use_deterministic_algorithms(previous)


def ordering_phase(v3: Path, output: Path) -> None:
    inputs = open_inputs(v3)
    native = _native_results(inputs, output)
    _run_deterministically(audit_prefix, inputs, output, [RankOrder.native("cuda")], native=True)
    _run_deterministically(audit_causality, inputs, output)
    published = output / "rank_order_manifest.json"
    orders = load_orders(output) if published.is_file() else calibrate(inputs, output)
    if any(json_at(published)[key] != value for key, value in inputs.provenance().items()):
        raise ValueError("Train-derived order does not match the current run")
    rows = run_budgets(inputs, NativeQuality(), output, orders[1:], "ordered")
    baseline, = json_at(output / "native_replay/metrics.json")
    result = progressive(native + rows, inputs.config, baseline)
    write_json(output / "progressive_quality.json", result)
    decision = decide_orders(inputs, output, orders, result)
    _run_deterministically(audit_prefix, inputs, output, orders[1:], native=False)
    inputs.finish(output / "ordered_run.json")
    print(f"Phase 1 static order selection: {decision}", flush=True)
