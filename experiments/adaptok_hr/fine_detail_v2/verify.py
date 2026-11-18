"""Re-read measured packets, source hashes, intervention rows and checkpoint files."""

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from adaptok_hr.codec.bitstream import HEADER, parse
from adaptok_hr.config import ModelConfig
from adaptok_hr.models.temporal_state import stream_identity
from adaptok_hr.runtime import file_hash, write_json
from cp_adaptok.metrics import summarize_latency

from .decisions import information, quality, result, select
from .evaluate import folder
from .run import CHECKPOINTS, PROTOCOL, ROOT
from .study import read


def jsonlines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def old_inputs():
    path = Path(PROTOCOL["historical_validation"]).parent / "closed_loop/k2048"
    return {(r["content_id"], r["frame"]): r["input_rgb_sha256"]
            for p in path.glob("*/frames.jsonl") for r in jsonlines(p)}


def protected_inputs():
    base = PROTOCOL["base_commit"]
    paths = ("cp_adaptok", "third_party", "configs/adaptok_hr", "experiments/adaptok_hr/fine_recovery",
             "docs/adaptok_hr_fine_recovery_report.md", "docs/adaptok_hr_fine_recovery_work_log.md")
    identities = {}
    for name in paths:
        old = subprocess.check_output(["git", "rev-parse", f"{base}:{name}"], text=True).strip()
        current = subprocess.check_output(["git", "rev-parse", f"HEAD:{name}"], text=True).strip()
        if old != current:
            raise ValueError(f"protected Git object changed: {name}")
        identities[name] = current
    for path in (Path("checkpoints/adaptok_hr/joint/step_001000.pt"),
                 Path("checkpoints/adaptok_hr_fine_recovery/A1/step_001000.pt"),
                 Path("checkpoints/adaptok_hr_fine_recovery/A3/step_001000.pt")):
        digest = file_hash(path)
        if digest != read(path.with_suffix(".json"))["checkpoint_sha256"]:
            raise ValueError(f"protected weights changed: {path}")
        identities[str(path)] = digest
    return identities


def verify_validation(path, originals):
    result = read(path / "summary.json")
    name = path.parent.name
    count = 0
    suffix_counts, frame_diversity = Counter(), {}
    metadata = read(CHECKPOINTS / name / f"step_{result['steps']:06d}.json")
    identity = {"config": ModelConfig(**metadata["config"]["model"]).to_dict(), "amp_bf16": True}
    wire_model = hashlib.sha256(bytes.fromhex(metadata["model_sha256"]) + json.dumps(identity, sort_keys=True).encode()).digest()
    if result["identity"]["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError(f"unmatched validation checkpoint {path}")
    for scope in ("closed_loop", "same_context"):
        previous = {}
        rows = jsonlines(path / f"{scope}.jsonl")
        budgets = Counter()
        for row in rows:
            k, frame, source = row["tokens"], row["frame"], row["content_id"]
            data = (folder(path, scope, k, source) / f"{frame:04d}.bin").read_bytes()
            packet = parse(data)
            digest = hashlib.sha256(data).digest()
            expected = (bytes(32) if frame % 32 == 0 else previous[source, k] if scope == "closed_loop"
                        else previous[source, 2048])
            if (packet.model != wire_model or packet.previous != expected or packet.codebook.hex() != metadata["codebook_sha256"]
                    or packet.stream_id != stream_identity(source) or packet.frame_index != frame
                    or len(packet.ids) != k or (packet.width, packet.height, packet.gop) != (1920, 1080, 32)
                    or row["bytes"] != len(data) or row["packet_sha256"] != digest.hex()
                    or originals[source, frame] != row["input_rgb_sha256"]):
                raise ValueError(f"incorrect packet/input/chain: {path} {scope} {source} {frame} {k}")
            if scope == "same_context":
                full = (folder(path, "closed_loop", 2048, source) / f"{frame:04d}.bin").read_bytes()
                if packet.ids != parse(full).ids[:k]:
                    raise ValueError("stored packet is not the exact canonical prefix")
                low, high = PROTOCOL["information"]["suffix"]
                if k == high:
                    suffix = packet.ids[low:high]
                    suffix_counts.update(suffix)
                    frame_diversity[source, frame] = len(set(suffix))
            previous[source, k] = digest
            count += 1
            budgets[k] += 1
        expected_frames = sum(frame < result["identity"]["frames_per_source"] for _, frame in originals)
        if set(budgets.values()) != {expected_frames} or set(budgets) != {256, 512, 1024, 2048}:
            raise ValueError("missing Validation frames or budgets")
    interventions = jsonlines(path / "suffix_interventions.jsonl")
    histogram = {i: n for i, n in enumerate(result["suffix"]["histogram"]) if n}
    if suffix_counts != histogram or any(frame_diversity[r["content_id"], r["frame"]] != r["within_frame_used"]
                                         for r in interventions):
        raise ValueError("reported suffix usage differs from actual transmitted IDs")
    if information(interventions) != result["information"]:
        raise ValueError("recorded decision disagrees with the raw interventions")
    return {"packets_reparsed": count, "interventions": len(interventions),
            "common_donor_frames": sum(r["donor"] is not None for r in interventions),
            "checkpoint_sha256": result["identity"]["checkpoint_sha256"], "input_hashes_match_legacy": True}


def verify_timing(row, budget):
    frames = row["frames"]
    total = len(frames)
    length = HEADER.size + (budget * 13 + 7) // 8
    flags = ("sender_receiver_bitwise_equal", "gop33_reset_bitwise_equal", "independent_model_instances", "amp_bf16")
    if total != 64 or not all(row[name] for name in flags):
        raise ValueError("incomplete native 64-frame independent endpoint measurement")
    for index, frame in enumerate(frames):
        if frame["frame"] != index or frame["frame_type"] != ("I" if index % 32 == 0 else "P"):
            raise ValueError("invalid causal GOP history in benchmark")
        if frame["tokens"] != budget or frame["bytes"] != length:
            raise ValueError("benchmark token/packet count mismatch")
        if not (0 < frame["encoder_ms"] < frame["sender_ms"] and frame["receiver_ms"] > 0 and
                math.isclose(frame["joint_ms"], frame["sender_ms"] + frame["receiver_ms"])):
            raise ValueError("joint latency omits part of the encoder or one decoder")
    if row["bytes"] != total * length or not math.isclose(row["actual_bpp"], length * 8 / (1080 * 1920)):
        raise ValueError("aggregate rate mismatch")
    for component, timing in row["timing"].items():
        actual = summarize_latency([r[component + "_ms"] for r in frames])
        if any(not math.isclose(actual[k], timing[k], rel_tol=1e-10) for k in actual):
            raise ValueError("mismatched aggregate endpoint throughput or latency")
    return {"frames": total, "bytes": row["bytes"], "joint_fps": row["timing"]["joint"]["fps"]}


def benchmarks():
    records = {}
    for path in sorted(ROOT.glob("S*/benchmark_*_*.json")):
        data = read(path)
        step = int(path.stem.split("_")[1])
        repeat = int(path.stem.split("_")[2])
        required_budgets = {"256", "512", "1024", "2048"} if repeat == 1 else {"2048"}
        metadata = read(CHECKPOINTS / path.parent.name / f"step_{step:06d}.json")
        if (not data["no_other_gpu_compute_process_before_and_after"] or data["completed_steps"] != step
                or set(data["budgets"]) != required_budgets or data["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or data["input"]["frame_indices"] != list(range(64))):
            raise ValueError(f"idle GPU or matched checkpoint/frames not verified: {path}")
        records[str(path.relative_to(ROOT))] = {k: verify_timing(v, int(k)) for k, v in data["budgets"].items()}
    return records


def verify_gates():
    first = read(ROOT / "decision.json")
    historical = read(PROTOCOL["historical_validation"])
    if first["winner"] != select() or first["new_test_used"]:
        raise ValueError("recorded screen or data selection disagrees with protocol")
    for seed, saved in first["seeds"].items():
        config = read(CHECKPOINTS / f"S{seed}_R00" / "step_000000.json")["config"]
        seeds = [config["training"]["model_seed"], config["experiment"]["data_seed"], config["training"]["budget_seed"]]
        if seeds != PROTOCOL["seeds"][int(seed) - 1]:
            raise ValueError("confirmation uses the wrong independently registered seeds")
        measured = result(f"S{seed}_{first['winner']}", 1000)
        control = result(f"S{seed}_R00", 1000)
        if saved != {"information": measured["information"], "quality": quality(measured, control, historical)}:
            raise ValueError("recorded confirmation fails replay")
    both = len(first["seeds"]) == 2 and all(r["information"]["pass"] for r in first["seeds"].values())
    if len(first["seeds"]) == 2:
        matches = [read(ROOT / f"match_S{s}_1000.json") for s in (1, 2)]
        for field in ("common_initial_full_weights", "schedule_sha256"):
            if set(matches[0][field].values()) & set(matches[1][field].values()):
                raise ValueError(f"seed replications share {field}")
    if first["decoder_stage_allowed"] != both or first["negative_information_stop"] == both:
        raise ValueError("conditional decoder eligibility disagrees with measured results")
    if not both:
        return {"representation": first, "decoder": None, "speed": None}
    from .decoder_stage import STOP, rank

    second = read(ROOT / "decoder_decision.json")
    best = min(second["seeds"]["1"], key=lambda arm: rank(arm, second["seeds"]["1"]))
    if second["decoder_winner"] != best:
        raise ValueError("decoder selection fails preregistered ranking")
    for seed, rows in second["seeds"].items():
        required = {"D0", "D1", "D2", "D3"} if seed == "1" else {"D0", best}
        if set(rows) != required:
            raise ValueError("required matched decoder arms missing")
        control = result(f"S{seed}_C0", STOP)
        for arm, saved in rows.items():
            measured = result(f"S{seed}_{arm}", STOP)
            if saved["quality"] != quality(measured, control, historical) or saved["information"] != measured["information"]["pass"]:
                raise ValueError("recorded decoder evidence fails replay")
    ready = all(rows[best]["quality"]["pass"] and rows[best]["information"] for rows in second["seeds"].values())
    parent = all(r["quality"]["pass"] for r in first["seeds"].values())
    if second["adoption_ready"] != ready or second["performance_optimization_allowed"] != (ready or parent):
        raise ValueError("speed-optimization eligibility disagrees with replicated quality")
    speed = read(ROOT / "performance_decision.json") if second["performance_optimization_allowed"] else None
    return {"representation": first, "decoder": second, "speed": speed}


def run():
    originals = old_inputs()
    validations = {str(p.parent.relative_to(ROOT)): verify_validation(p.parent, originals)
                   for p in sorted(ROOT.glob("*/validation_*_*/summary.json"))}
    weights = {}
    for meta in sorted(CHECKPOINTS.glob("*/*.json")):
        path = meta.with_suffix(".pt")
        observed = file_hash(path)
        if observed != read(meta)["checkpoint_sha256"]:
            raise ValueError(f"wrong checkpoint hash: {path}")
        weights[str(path)] = observed
    tests = ElementTree.parse(ROOT / "tests.xml").find(".//testsuite").attrib
    stages = verify_gates()
    mutations = read(ROOT / "mutations.json")
    audit = read(ROOT / "quality_audit.json")
    production = {"adaptok_hr/config.py", "adaptok_hr/models/tokenizer.py", "adaptok_hr/training/fine_detail.py",
                  "adaptok_hr/training/train.py", "adaptok_hr/models/decoder.py", "adaptok_hr/training/losses.py"}
    flagged = [x for x in audit["crap_over_10"] if x["file"] in production]
    timings = benchmarks()
    baseline_timings = {f"{p.parents[1].name}/benchmark_{p.parent.name.split('_')[1]}_1.json"
                        for p in ROOT.glob("S*/validation_*_*/summary.json")}
    checks = {"native_idle_gpu_timings": bool(timings) and baseline_timings.issubset(timings),
              "packets_and_histories": bool(validations), "checkpoint_files_rehashed": bool(weights),
              "tests": int(tests["tests"]) >= (89 if stages["decoder"] else 82)
                       and all(tests[k] == "0" for k in ("failures", "errors", "skipped")),
              "mutations": len(mutations["results"]) >= (34 if stages["decoder"] else 31) and mutations["all_killed"],
              "changed_production_complexity": not flagged, "duplicates": not audit["duplicates_over_08"]}
    record = {"engineering_checks_pass": all(checks.values()), "checks": checks, "validations": validations,
              "protected_git_and_checkpoints": protected_inputs(), "benchmark_runs": timings,
              "checkpoints": weights, "tests": tests, "mutations_killed": sum(r["killed"] for r in mutations["results"]),
              "flagged_production": flagged, "legacy_high_crap_count": len(audit["crap_over_10"]),
              "quality_and_scientific_gates": stages}
    write_json(ROOT / "acceptance.json", record)
    if not record["engineering_checks_pass"]:
        raise ValueError(checks)
    print(json.dumps({"checks": checks, "packets": sum(v["packets_reparsed"] for v in validations.values()), "weights": len(weights)}))


if __name__ == "__main__":
    run()
