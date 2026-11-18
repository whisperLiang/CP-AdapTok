"""Audit real checkpoints, optimizer counts, shared exposure and native packet records."""

import argparse
import json
from collections import Counter

import torch

from adaptok_hr.codec.bitstream import HEADER, parse
from adaptok_hr.codec.codec import tensor_digest
from adaptok_hr.config import ModelConfig
from adaptok_hr.models.model import AdapTokHR
from adaptok_hr.runtime import file_hash, stable_hash, write_json

from .evaluate import read_rows
from .run import ROOT, checkpoint


def optimizer_steps(saved):
    model = AdapTokHR(ModelConfig(**saved["config"]["model"]))
    group, = saved["optimizer"]["param_groups"]
    return {name: int(saved["optimizer"]["state"][key]["step"])
            for (name, _), key in zip(model.named_parameters(), group["params"], strict=True)
            if key in saved["optimizer"]["state"]}


def checkpoint_proof(name, step):
    path = checkpoint(name, step)
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["checkpoint_sha256"] != file_hash(path) or state["config_identity"] != stable_hash(state["config"]):
        raise ValueError("checkpoint integrity mismatch")
    if meta["model_sha256"] != tensor_digest(state["model"]).hex():
        raise ValueError("model tensor digest mismatch")
    current = optimizer_steps(state)
    start = 1300 if name.split("_", 1)[1].startswith("U") else 0
    if start:
        parent = torch.load(checkpoint(name, start), map_location="cpu", weights_only=False)
        baseline = optimizer_steps(parent)
        actual = {key: count - baseline.get(key, 0) for key, count in current.items()}
    else:
        actual = current
    if not actual or any(value != step - start for value in actual.values()):
        raise ValueError("Adam update counters differ from the declared optimizer exposure")
    rows = [r for r in read_rows(ROOT / name / "training.jsonl") if start < r["step"] <= step]
    if len(rows) != step - start or [r["step"] for r in rows] != list(range(start + 1, step + 1)):
        raise ValueError("optimizer log sequence differs")
    frames = [frame for row in rows for frame in row["frame_metrics"]]
    pairs = Counter(tuple(sorted((r["main_budget"], r["auxiliary_budget"]))) for r in frames if r["auxiliary_budget"] is not None)
    return {"checkpoint_sha256": meta["checkpoint_sha256"], "model_sha256": meta["model_sha256"],
            "optimizer_parameter_counters": actual, "source_sha256": meta["source_sha256"],
            "new_optimizer_updates": len(rows), "new_frames": len(frames),
            "paired_fraction": sum(pairs.values())/len(frames), "pairs": {str(k): v for k, v in pairs.items()},
            "crop_rng_checkpointed": "crop_rng" in state, "budget_rng_checkpointed": "budget_rng" in state}


def packet_proof(directory):
    closed = read_rows(directory / "closed_loop.jsonl")
    same = read_rows(directory / "same_context.jsonl")
    for scope, rows in (("closed_loop", closed), ("same_context", same)):
        for row in rows:
            path = directory / scope / f"K{row['tokens']}" / row["content_id"].replace(":", "_") / f"{row['frame']:04d}.bin"
            raw = path.read_bytes()
            packet = parse(raw)
            if (file_hash(path) != row["packet_sha256"] or len(raw) != row["bytes"]
                    or len(raw) != HEADER.size + (len(packet.ids)*13 + 7)//8):
                raise ValueError("packet hash or real 13-bit rate mismatch")
    high = {(r["content_id"], r["frame"]): r for r in same if r["tokens"] == 2048}
    for row in same:
        source = high[row["content_id"], row["frame"]]
        if row["input_rgb_sha256"] != source["input_rgb_sha256"]:
            raise ValueError("same-state budgets used different native input")
        folder = directory / "same_context"
        relative = row["content_id"].replace(":", "_") + f"/{row['frame']:04d}.bin"
        low = parse((folder / f"K{row['tokens']}" / relative).read_bytes())
        full = parse((folder / "K2048" / relative).read_bytes())
        if low.ids != full.ids[:row["tokens"]] or low.previous != full.previous:
            raise ValueError("nested IDs or shared causal context differ")
    return {"closed_loop_packets": len(closed), "same_state_packets": len(same),
            "all_packet_hashes_and_13bit_sizes_verified": True, "all_nested_prefixes_and_histories_verified": True,
            "input_schedule_sha256": stable_hash([(r["content_id"], r["frame"], r["tokens"], r["input_rgb_sha256"]) for r in closed])}


def verify():
    results = {}
    schedules = {}
    for summary in sorted(ROOT.glob("S*/validation_*_64/summary.json")):
        name = summary.parents[1].name
        step = int(summary.parent.name.split("_")[1])
        proof = {"checkpoint": checkpoint_proof(name, step), "packets": packet_proof(summary.parent)}
        results[f"{name}/{step}"] = proof
        schedules.setdefault((name[1], name.split("_")[1][0], step), []).append(proof["packets"]["input_schedule_sha256"])
    if any(len(set(values)) != 1 for values in schedules.values()):
        raise ValueError("Validation inputs differ across matched arms")
    write_json(ROOT / "acceptance.json", {"executed_runs": results, "matching_validation_inputs": True})
    print(json.dumps({"verified_runs": len(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify()
