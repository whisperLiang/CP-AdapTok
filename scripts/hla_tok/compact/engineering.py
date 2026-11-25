"""Capture the regression evidence and verify historical and Train asset identity."""

import subprocess
from pathlib import Path
from xml.etree import ElementTree

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from scripts.hla_tok.compact.inputs import CONFIG, OUTPUT, TRAIN_PLAN
from scripts.hla_tok.continuous.inputs import run_arguments
from scripts.hla_tok.v2.protocol import ROOT, json_at
from scripts.hla_tok.v3.protocol import read_config

V3_PATHS = ("results/hla_tok/v3", "docs/hla_tok/v3", "configs/hla_tok/v3", "checkpoints/hla_tok/v3")
C_PATHS = ("results/hla_tok/continuous", "docs/hla_tok/continuous", "configs/hla_tok/continuous")
SEALED = (
    "reproduction", "audit_inputs", "projection_inputs", "scalar_inputs", "checks_inputs", "training_inputs",
    "frame_reference_inputs", "original_reference_inputs", "benchmark_inputs", "sampling_inputs", "lossless_audit",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot(v3: Path, continuous: Path) -> dict:
    repos = {"continuous": (continuous, C_PATHS), "v3": (v3, V3_PATHS)}
    records = {}
    for name, (repo, folders) in repos.items():
        if _git(repo, "status", "--porcelain"):
            raise ValueError(f"frozen historical worktree is dirty: {repo}")
        files = [path for folder in folders for path in sorted((repo / folder).rglob("*")) if path.is_file()]
        records[name] = {"commit": _git(repo, "rev-parse", "HEAD"), "files": len(files),
                         "sha256": {str(path.relative_to(repo)): sha256_file(path) for path in files}}
    return records


def _same_historical_source(base: str) -> bool:
    paths = [*C_PATHS, *V3_PATHS[:3], "data/manifests", "third_party",
             "hla_tok/models", "hla_tok/progressive", "hla_tok/streaming/continuous.py",
             "hla_tok/streaming/continuous_wire.py", "scripts/hla_tok/continuous", "tests/hla_tok/continuous"]
    return not _git(ROOT, "diff", base, "--", *paths)


def historical(v3: Path, continuous: Path, output: Path, *, before: bool) -> None:
    report = snapshot(v3, continuous)
    path = output / "historical_assets_before.json"
    if before:
        if path.exists() and json_at(path) != report:
            raise ValueError("historical assets differ from the initial snapshot")
        write_json(path, report)
        return
    config = read_config(CONFIG)
    training = json_at(output / "training_selection.json")
    stages = SEALED + (("trained_checks_inputs",) if training["adopted"] else ())
    counts = {stage: json_at(output / f"{stage}.json")["test_pixel_reads"] for stage in stages}
    if report != json_at(path) or not _same_historical_source(config["base_commit"]) or any(counts.values()):
        raise ValueError("historical or Test sentinel differs")
    profiles = json_at(output / "compact_profile_manifest.json")
    if training["adopted"]:
        profiles[training["asset"]] = training["spec"]
    observed = {key: sha256_file(output / "profiles" / asset["filename"]) for key, asset in profiles.items()}
    if observed != {key: asset["sha256"] for key, asset in profiles.items()}:
        raise ValueError("published Compact Profile assets differ")
    c0 = json_at(output / "checkpoint_manifest.json")
    if any(json_at(output / f"{name}.json")["state_dict_after_sha256"] != c0["state_dict_sha256"]
           for name in stages if name.endswith("_inputs")):
        raise ValueError("C0 model weights changed")
    write_json(output / "historical_assets_audit.json", {
        "status": "PASS", "historical": report, "test_pixel_reads": sum(counts.values()), "test_reads_by_stage": counts,
        "no_diff_from_frozen_c0_source": True, "old_v3_fixed13_formal_verdict": "FAIL",
        "c0_checkpoint_sha256": c0["sha256"], "c0_state_sha256": c0["state_dict_sha256"],
        "compact_asset_sha256": observed, "train_plan_sha256": sha256_file(TRAIN_PLAN),
        "train_cache_manifest_sha256": sha256_file(output / "cache/identity.json"), "protocol_sha256": sha256_file(CONFIG),
    })


def _pytest_results(path: Path) -> dict:
    tree = ElementTree.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
    stats = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    good = stats["tests"] > stats["skipped"] and stats["errors"] == stats["failures"] == 0
    return stats | {"status": "PASS" if good else "FAIL"}


def attest(output: Path) -> None:
    audit = {name: _pytest_results(output / "logs" / (name + ".xml"))
             for name in ("cpu_tests", "cuda_tests", "cross_device")}
    tools = json_at(output / "logs/tool_status.json")
    audit.update(tools)
    audit["targeted_mutations"] = json_at(output / "targeted_mutations.json")
    write_json(output / "engineering_audit.json", audit)


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    parser.add_argument("action", choices=("before", "after", "attest"))
    parser.add_argument("--continuous", type=Path, default=ROOT.parent.parent / "hla-tok-continuous" / ROOT.name)
    args = parser.parse_args()
    if args.action == "attest":
        attest(args.output)
    else:
        if args.continuous == ROOT:
            raise ValueError("historical worktree must be distinct")
        historical(args.v3, args.continuous, args.output, before=args.action == "before")


if __name__ == "__main__":
    main()
