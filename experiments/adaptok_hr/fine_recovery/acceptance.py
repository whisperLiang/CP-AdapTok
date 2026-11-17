"""Validate measured artifacts and keep engineering acceptance separate from research success."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.runtime import file_hash, write_json

ROOT = Path("results/adaptok_hr_fine_recovery")


def read(path):
    return json.loads(Path(path).read_text())


def test_evidence():
    tree = ET.parse(ROOT / "tests_final.xml").getroot()
    return {"tests": len(list(tree.iter("testcase"))), "failures": len(list(tree.iter("failure"))),
            "errors": len(list(tree.iter("error"))), "skipped": len(list(tree.iter("skipped")))}


def verify_checkpoints():
    result = {}
    for path in sorted(Path("checkpoints/adaptok_hr_fine_recovery").rglob("step_*.json")):
        metadata = read(path)
        sha = file_hash(path.with_suffix(".pt"))
        if sha != metadata["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        result[str(path.with_suffix(".pt"))] = sha
    return result


def verify_packets(directory, budget, codebook_sha):
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    files = sorted(directory.glob("frame_*.bin"))
    if len(files) != len(frames):
        raise ValueError("missing actual packet")
    for path, row in zip(files, frames):
        data = path.read_bytes()
        packet = parse(data)
        if (len(data), len(packet.ids), packet.frame_index, packet.codebook.hex()) != (
                row["bytes"], budget, row["frame"], codebook_sha):
            raise ValueError(f"packet accounting or identity mismatch: {path}")
    return len(frames), sum(p.stat().st_size for p in files)


def verify_validation(name):
    directory = ROOT / name / "validation_001000_64f"
    r = read(directory / "summary.json")
    checkpoint = read(f"checkpoints/adaptok_hr_fine_recovery/{name}/step_001000.json")
    if r["protocol"]["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
        raise ValueError("Validation is not from selected checkpoint")
    if r["protocol"]["selection_split"] != "validation":
        raise ValueError("model selection must never use Test")
    result = {}
    for key, summary in r["closed_loop"].items():
        counts = [verify_packets(d, int(key), checkpoint["codebook_sha256"])
                  for d in sorted((directory / f"closed_loop/k{key}").iterdir())]
        if (sum(c[0] for c in counts), sum(c[1] for c in counts)) != (summary["frames"], summary["bytes"]):
            raise ValueError("global packet accounting mismatch")
        result[key] = {"frames": summary["frames"], "bytes": summary["bytes"], "bpp": summary["actual_bpp"]}
    for path in (directory / "same_context_packets").glob("*/frame_*_K2048.bin"):
        full = parse(path.read_bytes())
        for budget in (256, 512, 1024, 2048):
            shorter = path.with_name(path.name.replace("K2048", f"K{budget}"))
            if parse(shorter.read_bytes()).ids != full.ids[:budget]:
                raise ValueError("stored same-context packet prefix mismatch")
    return result


def verify_benchmarks(selected):
    result = {}
    for name in selected:
        r = read(ROOT / name / "benchmark_001000.json")
        result[name] = {}
        for budget, value in r["budgets"].items():
            if len(value["frames"]) != 64 or not value["sender_receiver_bitwise_equal"]:
                raise ValueError("native 64-frame endpoint validation missing")
            if not value["gop33_reset_bitwise_equal"] or not r["no_other_gpu_compute_process_before_and_after"]:
                raise ValueError("reset check or independent throughput precondition missing")
            result[name][budget] = value["timing"]["joint"]["fps"]
    return result


def verify_input_parity(selected):
    directory = ROOT / "baseline/validation_001000_64f/closed_loop"
    expected = {}
    for path in directory.glob("k*/*/frames.jsonl"):
        key = str(path.relative_to(directory))
        expected[key] = [json.loads(s)["input_rgb_sha256"] for s in path.read_text().splitlines()]
    if not expected:
        raise ValueError("baseline input parity evidence missing")
    for name in selected:
        for key, hashes in expected.items():
            path = ROOT / name / "validation_001000_64f/closed_loop" / key
            actual = [json.loads(s)["input_rgb_sha256"] for s in path.read_text().splitlines()]
            if actual != hashes:
                raise ValueError("Validation input pixels/order differ from baseline")
    return {"same_native_rgb_and_frame_order": True, "models": ["baseline", *selected],
            "budget_source_sequences": len(expected), "frames_each_budget": sum(len(v) for k, v in expected.items() if k.startswith("k2048/"))}


def main():
    selected = read(ROOT / "selection_300.json")["selected"]
    tests, mutations = test_evidence(), read(ROOT / "mutations.json")
    scoped_audit = read(ROOT / "quality_audit_changed.json")
    matching = [read(ROOT / f"matching_{step:06d}.json") for step in (300, 1000)]
    protected = read(ROOT / "protected_inputs.json")
    validation = {n: verify_validation(n) for n in selected}
    benchmark = verify_benchmarks(selected)
    result = {"tests": tests, "mutations": len(mutations["results"]), "mutations_all_killed": mutations["all_killed"],
              "changed_production_crap_over_10": scoped_audit["crap_over_10"],
              "duplicate_functions_over_08": scoped_audit["duplicates_over_08"],
              "matching": matching, "protected_inputs": protected, "validation_actual_packets": validation,
              "validation_input_parity": verify_input_parity(selected),
              "independent_native_64_frame_joint_fps": benchmark, "checkpoint_sha256": verify_checkpoints(),
              "research_decision": read(ROOT / "decision.json"),
              "engineering_checks_pass": not any(tests[k] for k in ("failures", "errors", "skipped"))
                and mutations["all_killed"] and not scoped_audit["crap_over_10"] and not scoped_audit["duplicates_over_08"]}
    write_json(ROOT / "acceptance.json", result)
    print(json.dumps({k: result[k] for k in ("tests", "mutations_all_killed", "engineering_checks_pass", "research_decision")}))


if __name__ == "__main__":
    main()
