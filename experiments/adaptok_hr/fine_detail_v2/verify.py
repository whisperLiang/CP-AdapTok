"""Re-read measured packets, source hashes, intervention rows and checkpoint files."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.models.temporal_state import stream_identity
from adaptok_hr.runtime import file_hash, write_json

from .decisions import information
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
            if (packet.previous != expected or packet.codebook.hex() != metadata["codebook_sha256"]
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
        expected_frames = sum(min(result["identity"]["frames_per_source"], n) for n in (50, 64, 60))
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
    mutations = read(ROOT / "mutations.json")
    audit = read(ROOT / "quality_audit.json")
    production = {"adaptok_hr/config.py", "adaptok_hr/models/tokenizer.py", "adaptok_hr/training/fine_detail.py",
                  "adaptok_hr/training/train.py", "adaptok_hr/models/decoder.py", "adaptok_hr/training/losses.py"}
    flagged = [x for x in audit["crap_over_10"] if x["file"] in production]
    checks = {"packets_and_histories": bool(validations), "checkpoint_files_rehashed": bool(weights),
              "tests": int(tests["tests"]) >= 82 and all(tests[k] == "0" for k in ("failures", "errors", "skipped")),
              "mutations": len(mutations["results"]) >= 31 and mutations["all_killed"],
              "changed_production_complexity": not flagged, "duplicates": not audit["duplicates_over_08"]}
    record = {"engineering_checks_pass": all(checks.values()), "checks": checks, "validations": validations,
              "protected_git_and_checkpoints": protected_inputs(),
              "checkpoints": weights, "tests": tests, "mutations_killed": sum(r["killed"] for r in mutations["results"]),
              "flagged_production": flagged, "legacy_high_crap_count": len(audit["crap_over_10"]),
              "quality_and_scientific_gates": read(ROOT / "decision.json")}
    write_json(ROOT / "acceptance.json", record)
    if not record["engineering_checks_pass"]:
        raise ValueError(checks)
    print(json.dumps({"checks": checks, "packets": sum(v["packets_reparsed"] for v in validations.values()), "weights": len(weights)}))


if __name__ == "__main__":
    run()
