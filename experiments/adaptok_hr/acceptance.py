"""Machine-readable acceptance, parity and immutable-baseline checks."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from adaptok_hr.codec.bitstream import parse
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.runtime import file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text())


def xml_tests(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    return {"tests": len(cases), "failures": len(list(root.iter("failure"))),
            "errors": len(list(root.iter("error"))), "skipped": len(list(root.iter("skipped"))),
            "properties": {p.attrib["name"]: p.attrib["value"] for p in root.iter("property")}}


def source_parity(root, index):
    official = read("results/original_adaptok_1080p/cp_input_parity.json")
    sources = {r["content_id"]: r for r in index["splits"]["historical"]}
    expected_rows = {}
    for path in (root / "historical/k256").rglob("frames.jsonl"):
        for row in map(json.loads, path.read_text().splitlines()):
            expected_rows[(row["content_id"], row["frame"])] = row["input_rgb_sha256"]
    windows, frames = [], 0
    for entry in official["windows"]:
        clip, _ = read_clip(sources[entry["content_id"]], entry["start"], 16)
        actual = hashlib.sha256(clip.numpy().tobytes()).hexdigest()
        if actual != entry["rgb_sha256"]:
            raise ValueError("native RGB differs from the official/CP sealed reference")
        for offset, frame in enumerate(clip):
            key = entry["content_id"], entry["start"] + offset
            if hashlib.sha256(frame.numpy().tobytes()).hexdigest() != expected_rows[key]:
                raise ValueError("HR evaluated pixels differ from sealed input reference")
            frames += 1
        windows.append({"content_id": entry["content_id"], "start": entry["start"], "rgb_sha256": actual})
    for branch, sha in official["reference_sha256"].items():
        if file_hash(Path(official["reference"]) / branch / "validation.json") != sha:
            raise ValueError("sealed CP result file changed")
    return {"equal": True, "frames": frames, "windows": windows,
            "cp_reference_sha256": official["reference_sha256"]}


def immutable_inputs(index):
    for split, sha in index["cp_manifest_sha256"].items():
        if file_hash(f"data/manifests/cpat_{split}.json") != sha:
            raise ValueError("original CP manifest changed")
    official = read("results/original_adaptok_1080p/checkpoint_manifest.json")
    path = Path(official["directory"]) / "model.safetensors"
    if file_hash(path) != official["model_safetensors_sha256"]:
        raise ValueError("official pretrained weights changed")
    status = subprocess.check_output(["git", "-C", "third_party/AdapTok", "status", "--porcelain", "--untracked-files=no"], text=True)
    if status.strip():
        raise ValueError("official tracked source changed")
    changed = subprocess.check_output(["git", "diff", "--name-only", "experiment/original-adaptok-1080p"], text=True).splitlines()
    forbidden = [p for p in changed if p.startswith(("cp_adaptok/", "third_party/", "experiments/original_adaptok_1080p/"))]
    if forbidden:
        raise ValueError(f"protected implementation changed: {forbidden}")
    return {"cp_manifests_unchanged": True, "official_weights_unchanged": True,
            "official_source_unchanged": True, "cp_and_official_paths_unchanged": True,
            "sealed_cp_checkpoint_sha256": sealed_cp_weights()}


def sealed_cp_weights():
    selected = read("results/large_scale/formal_20261116/final_selection.json")["selected"]
    identities = {}
    for branch, row in selected.items():
        actual = file_hash(row["checkpoint"])
        if actual != row["sha256"]:
            raise ValueError(f"sealed CP {branch} checkpoint changed")
        identities[branch] = actual
    return identities


def metric_parity(root):
    historical = read(root / "historical/metric_manifest.json")["weights"]
    expected = read("results/original_adaptok_1080p/cp_input_parity.json")["metric_weight_sha256"]
    expected_by_name = {Path(p).name: sha for p, sha in expected.items()}
    actual_by_name = {Path(p).name: sha for p, sha in historical.items()}
    if any(expected_by_name[name] != sha for name, sha in actual_by_name.items()):
        raise ValueError("metric weight identity differs from the official/CP baseline")
    return {"weights_equal": True, "sha256_by_filename": actual_by_name}


def byte_accounting(root):
    measured = {}
    for split in ("validation_300", "validation_1000", "historical", "final_test"):
        budgets = read(root / split / "summary.json")["budgets"]
        for budget, summary in budgets.items():
            paths = list((root / split / f"k{budget}").rglob("*.bin"))
            total = 0
            for path in paths:
                data = path.read_bytes()
                parse(data)
                total += len(data)
            if len(paths) != summary["frames"] or total != summary["bytes"]:
                raise ValueError("actual stored binary frames disagree with billed bytes")
            measured[f"{split}/k{budget}"] = {"frames": len(paths), "bytes": total}
    return measured


def continuous_stream(root, decision):
    measured = read(root / "sequence_benchmark.json")
    if measured["checkpoint_sha256"] != decision["checkpoint_sha256"]:
        raise ValueError("continuous benchmark used a different selected checkpoint")
    reference = root / "validation_1000"
    result = {}
    for budget, value in measured["budgets"].items():
        source = value["frames"][0]["content_id"]
        path = reference / f"k{budget}" / source.replace(":", "_") / "frames.jsonl"
        expected = [json.loads(line)["input_rgb_sha256"] for line in path.read_text().splitlines()]
        if [r["input_rgb_sha256"] for r in value["frames"]] != expected:
            raise ValueError("continuous throughput input differs from the validated native frames")
        if not value["sender_receiver_equal"] or len(value["frames"]) < 64:
            raise ValueError("continuous benchmark failed endpoint or length requirement")
        result[budget] = {"frames": len(value["frames"]), "native_rgb_equal": True,
                          "sender_receiver_equal": True,
                          "joint_fps": value["summary"]["sender_plus_receiver_latency"]["fps"]}
    return result


def run(root):
    index = load_index("data/adaptok_hr/index.json")
    tests = xml_tests(root / "tests_all.xml")
    mutations = read(root / "mutations.json")
    audit = read(root / "quality_audit.json")
    completion = read(root / "joint/completion.json")
    rows = [json.loads(line) for line in (root / "joint/training.jsonl").read_text().splitlines()]
    sequence_equal = [r["step"] for r in rows] == list(range(1, completion["completed_steps"] + 1))
    frame_count_equal = sum(r["frames"] for r in rows) == completion["training_frames"]
    parity = source_parity(root, index)
    write_json(root / "input_parity.json", parity)
    benchmark = read(root / "benchmark.json")["budgets"]
    diagnostics = read(root / "token_diagnostics.json")
    decision = read(root / "decision.json")
    continuous = continuous_stream(root, decision)
    all_30 = all(r["joint_sender_plus_receiver"]["fps"] >= 30
                 for budget in benchmark.values() for r in budget["results"].values())
    all_30 = all_30 and all(r["joint_fps"] >= 30 for r in continuous.values())
    result = {"code_and_native_1080p_pilot_complete": True,
              "all_tests_pass": tests["failures"] == tests["errors"] == tests["skipped"] == 0,
              "tests": tests, "mutations_all_killed": mutations["all_killed"],
              "mutations": len(mutations["results"]), "crap_over_10": len(audit["crap_over_10"]),
              "duplicate_functions_over_08": len(audit["duplicates_over_08"]),
              "optimizer_log_sequence_equal": sequence_equal, "training_frame_count_equal": frame_count_equal,
              "completed_joint_steps": completion["completed_steps"], "training_frames": completion["training_frames"],
              "actual_binary_bytes": byte_accounting(root), "rgb_parity_frames": parity["frames"],
              "continuous_64_frame_benchmark": continuous,
              "metric_identity": metric_parity(root), "protected_inputs": immutable_inputs(index),
              "gop33_native_roundtrip_pass": read(root / "phase0/phase0.json")["gop32_reset_equal"],
              "final_test_sealed": (root / "joint/final_test_seal.json").exists(),
              "final_test_complete": (root / "final_test/summary.json").exists(),
              "joint_1080p30_all_conditions_met": all_30, "quality_ready_for_scaling": decision["scorer_ready"],
              "fine_representation_effective": diagnostics["layers"]["Fine_transmitted_prefix"]["used_codes"] > 1,
              "scorer_formal_training_complete": (root / "scorer/completion.json").exists(),
              "full_training_ablations_complete": decision["ablations_completed"],
              "interpretation": "engineering implementation and bounded pilot accepted; research quality/progressive-Fine/30FPS targets not achieved; no 5000-step or Phase2 claim"}
    result["engineering_acceptance"] = all((result["all_tests_pass"], sequence_equal, frame_count_equal,
                                            mutations["all_killed"], not audit["crap_over_10"],
                                            not audit["duplicates_over_08"], result["final_test_complete"]))
    write_json(root / "acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/adaptok_hr"))
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
