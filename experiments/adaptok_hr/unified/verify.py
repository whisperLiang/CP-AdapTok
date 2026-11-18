"""Audit real checkpoints, optimizer counts, shared exposure and native packet records."""

import argparse
import json
from collections import Counter

import torch
from PIL import Image

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


def verified_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["checkpoint_sha256"] != file_hash(path) or state["config_identity"] != stable_hash(state["config"]):
        raise ValueError("checkpoint integrity mismatch")
    if meta["model_sha256"] != tensor_digest(state["model"]).hex():
        raise ValueError("model tensor digest mismatch")
    return state, meta


def verified_optimizer_counts(name, state, start, step):
    current = optimizer_steps(state)
    if start:
        parent = torch.load(checkpoint(name, start), map_location="cpu", weights_only=False)
        baseline = optimizer_steps(parent)
        actual = {key: count - baseline.get(key, 0) for key, count in current.items()}
    else:
        actual = current
    if not actual or any(value != step - start for value in actual.values()):
        raise ValueError("Adam update counters differ from the declared optimizer exposure")
    return actual


def verified_training_rows(name, start, step):
    rows = [r for r in read_rows(ROOT / name / "training.jsonl") if start < r["step"] <= step]
    if len(rows) != step - start or [r["step"] for r in rows] != list(range(start + 1, step + 1)):
        raise ValueError("optimizer log sequence differs")
    return rows



def scratch_initialization(name, state):
    initial, meta = verified_checkpoint(checkpoint(name, 0))
    if initial["completed_steps"] != 0 or initial["optimizer"]["state"]:
        raise ValueError("scratch initialization contains optimizer training history")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(state["config"]["training"]["model_seed"])
        model = AdapTokHR(ModelConfig(**state["config"]["model"]))
    expected = tensor_digest(model.state_dict()).hex()
    if meta["model_sha256"] != expected:
        raise ValueError("scratch checkpoint does not match its declared independent initialization")
    return expected


def checkpoint_proof(name, step):
    state, meta = verified_checkpoint(checkpoint(name, step))
    start = 1300 if name.split("_", 1)[1].startswith("U") else 0
    initial = scratch_initialization(name, state) if start == 0 else None
    actual = verified_optimizer_counts(name, state, start, step)
    rows = verified_training_rows(name, start, step)
    frames = [frame for row in rows for frame in row["frame_metrics"]]
    pairs = Counter(tuple(sorted((r["main_budget"], r["auxiliary_budget"]))) for r in frames if r["auxiliary_budget"] is not None)
    return {"checkpoint_sha256": meta["checkpoint_sha256"], "model_sha256": meta["model_sha256"],
            "optimizer_parameter_counters": actual, "source_sha256": meta["source_sha256"],
            "new_optimizer_updates": len(rows), "new_frames": len(frames), "scratch_initial_model_sha256": initial,
            "paired_fraction": sum(pairs.values())/len(frames), "pairs": {str(k): v for k, v in pairs.items()},
            "crop_rng_checkpointed": "crop_rng" in state, "budget_rng_checkpointed": "budget_rng" in state}


def verify_packet_rows(directory, scope, rows):
    for row in rows:
        path = directory / scope / f"K{row['tokens']}" / row["content_id"].replace(":", "_") / f"{row['frame']:04d}.bin"
        raw = path.read_bytes()
        packet = parse(raw)
        if (file_hash(path) != row["packet_sha256"] or len(raw) != row["bytes"]
                or len(raw) != HEADER.size + (len(packet.ids)*13 + 7)//8):
            raise ValueError("packet hash or real 13-bit rate mismatch")


def verify_nested_rows(directory, rows):
    high = {(r["content_id"], r["frame"]): r for r in rows if r["tokens"] == 2048}
    for row in rows:
        source = high[row["content_id"], row["frame"]]
        if row["input_rgb_sha256"] != source["input_rgb_sha256"]:
            raise ValueError("same-state budgets used different native input")
        folder = directory / "same_context"
        relative = row["content_id"].replace(":", "_") + f"/{row['frame']:04d}.bin"
        low = parse((folder / f"K{row['tokens']}" / relative).read_bytes())
        full = parse((folder / "K2048" / relative).read_bytes())
        if low.ids != full.ids[:row["tokens"]] or low.previous != full.previous:
            raise ValueError("nested IDs or shared causal context differ")


def packet_proof(directory):
    closed = read_rows(directory / "closed_loop.jsonl")
    same = read_rows(directory / "same_context.jsonl")
    verify_packet_rows(directory, "closed_loop", closed)
    verify_packet_rows(directory, "same_context", same)
    verify_nested_rows(directory, same)
    images = native_image_proof(directory / "samples")
    return {"native_images": images, "closed_loop_packets": len(closed), "same_state_packets": len(same),
            "all_packet_hashes_and_13bit_sizes_verified": True, "all_nested_prefixes_and_histories_verified": True,
            "input_schedule_sha256": stable_hash([(r["content_id"], r["frame"], r["tokens"], r["input_rgb_sha256"]) for r in closed])}



def native_image_proof(directory):
    result = {}
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as frame:
            if frame.size != (1920, 1080):
                raise ValueError("native sample image shape mismatch")
        result[path.name] = file_hash(path)
    return {"scope": "actual ID prefixes on common real K2048 history; I frames reset context; 8-bit display exports",
            "resolution": [1920, 1080], "files_sha256": result}


def verify():
    torch.set_num_threads(4)
    results = {}
    schedules = {}
    for summary in sorted(ROOT.glob("S*/validation_*_64/summary.json")):
        name = summary.parents[1].name
        step = int(summary.parent.name.split("_")[1])
        proof = {"checkpoint": checkpoint_proof(name, step), "packets": packet_proof(summary.parent)}
        results[f"{name}/{step}"] = proof
        schedules.setdefault((name[1], name.split("_")[1][0], step), []).append(proof["packets"]["input_schedule_sha256"])
    if len(results) < 8 or not (ROOT / "decision_A.json").exists():
        raise ValueError("the two-seed Pilot is not completely evaluated")
    if any(len(set(values)) != 1 for values in schedules.values()):
        raise ValueError("Validation inputs differ across matched arms")
    write_json(ROOT / "acceptance.json", {"executed_runs": results, "matching_validation_inputs": True})
    print(json.dumps({"verified_runs": len(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify()
