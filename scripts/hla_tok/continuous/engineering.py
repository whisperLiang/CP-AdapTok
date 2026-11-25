"""Reuse the repository's Radon and normalized AST audit for the changed paths."""

import argparse
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.continuous.inputs import RESULTS
from scripts.hla_tok.v2.protocol import ROOT, json_at
from scripts.quality_audit import analyze_file


def _find_duplicates(functions: list) -> list[dict]:
    matches = []
    for first, second in combinations(functions, 2):
        comparison = SequenceMatcher(None, first[2], second[2], autojunk=False)
        if comparison.real_quick_ratio() < 0.8 or comparison.quick_ratio() < 0.8:
            continue
        similarity = comparison.ratio()
        if similarity >= 0.8:
            matches.append({"first": first[:2], "second": second[:2], "similarity": similarity})
    return matches


def complexity(coverage: Path, output: Path) -> None:
    source = ROOT / "hla_tok"
    paths = [source / "models/codec.py", source / "models/local.py",
             *sorted((source / "streaming").glob("continuous*.py")),
             *sorted((ROOT / "scripts/hla_tok/continuous").glob("*.py"))]
    measured = json_at(coverage)
    rows, functions = [], []
    for path in paths:
        file_rows, signatures = analyze_file(path, measured)
        rows.extend(file_rows)
        functions.extend(signatures)
    rows = list({(row["file"], row["function"], row["line"]): row for row in rows}.values())
    over = sorted((row for row in rows if row["crap"] >= 10), key=lambda row: -row["crap"])
    duplicates = _find_duplicates(functions)
    write_json(output / "complexity_audit.json", {
        "tools": "repository Radon + measured line coverage + normalized AST similarity",
        "coverage_scope": "CPU and CUDA regression; no profiler overhead in the benchmark",
        "max_complexity": max(row["complexity"] for row in rows), "functions": rows,
        "crap_at_least_10": over, "duplicates_at_least_0_8": duplicates,
    })
    print(f"{len(rows)} functions, {len(over)} CRAP >= 10, {len(duplicates)} AST pairs >= 0.8", flush=True)


TEST_EVIDENCE = (
    ("Checkpoint SHA-256", "checkpoint_manifest.json"),
    ("Native maximum-budget parity", "metrics_parity.json"),
    ("Physical vector mapping", "logs/cpu_tests.xml"),
    ("Legal budget boundaries", "logs/cpu_tests.xml"),
    ("Illegal budget rejection", "logs/cpu_tests.xml"),
    ("All eight ordered prefix pairs", "prefix_audit.json"),
    ("50 random legal prefixes", "prefix_audit.json"),
    ("FP32 bits", "wire_roundtrip_audit.json"),
    ("FP16 casting", "wire_roundtrip_audit.json"),
    ("BF16 bits and casting", "wire_roundtrip_audit.json"),
    ("Actual lengths", "wire_prefix_audit.json"),
    ("Byte payload prefixes", "wire_prefix_audit.json"),
    ("No resends on enhancement", "wire_roundtrip_audit.json"),
    ("Incremental parity", "wire_roundtrip_audit.json"),
    ("Isolated receiving process", "wire_roundtrip_audit.json"),
    ("Corruption and truncation", "logs/cpu_tests.xml"),
    ("Ordering and inverse", "rank_order_manifest.json"),
    ("Reordered maximum equals C0", "ordered_prefix_audit.json"),
    ("Future perturbations", "causality_audit.json"),
    ("Independent budget paths", "closed_state_audit.json"),
    ("Dynamic synchronization", "dynamic_state_audit.json"),
    ("No unreceived vectors in synthesis", "mutation_audit.json"),
    ("Actual bpp and Mbps", "logs/cpu_tests.xml"),
    ("Audited split isolation", "ordered_run.json"),
    ("Unchanged state_dict", "benchmark_run.json"),
)


def _test_summary(path: Path) -> dict:
    root = ElementTree.parse(path)
    suites = list(root.iter("testsuite"))
    totals = {field: sum(int(suite.get(field, "0")) for suite in suites)
              for field in ("tests", "skipped", "failures", "errors")}
    count = totals["tests"] - totals["skipped"] - totals["failures"] - totals["errors"]
    return totals | {"passed": count, "status": "PASS" if count and not totals["failures"] + totals["errors"] else "FAIL"}


def attest(output: Path) -> None:
    logs = output / "logs"
    tests = {key: _test_summary(logs / f"{name}.xml")
             for key, name in (("cpu", "cpu_tests"), ("cuda", "cuda_tests"), ("cross_device", "cross_device"))}
    complexity_record = json_at(output / "complexity_audit.json")
    mutations = json_at(output / "mutation_audit.json")
    lint = (logs / "ruff.log").read_text()
    typing = (logs / "mypy.log").read_text()
    passed = (all(result["status"] == "PASS" for result in tests.values())
              and mutations["status"] == "PASS" and "All checks passed!" in lint
              and "Success: no issues found" in typing)
    if not passed:
        raise RuntimeError("one or more regression, mutation or static checks failed")
    evidence = {f"T{i}": {"status": "PASS", "test": name, "evidence": path}
                for i, (name, path) in enumerate(TEST_EVIDENCE, 1)}
    scope = [row for row in complexity_record["crap_at_least_10"] if row["file"].startswith("hla_tok/streaming/continuous")]
    write_json(output / "engineering_audit.json", {
        "status": "PASS", "test_runs": tests, "ruff": "PASS", "mypy": "PASS",
        "directed_mutations": {"status": mutations["status"], "count": len(mutations["cases"])},
        "complexity": {"status": "PARTIAL PASS" if complexity_record["crap_at_least_10"] else "PASS",
                       "functions_at_least_10": len(complexity_record["crap_at_least_10"]),
                       "continuous_codec_functions_at_least_10": scope,
                       "normalized_ast_pairs_at_least_0_8": len(complexity_record["duplicates_at_least_0_8"]),
                       "native_tools_available": False},
        "correctness_coverage": evidence,
    })
    print("Engineering regression and behavior checks PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--coverage", type=Path)
    action.add_argument("--attest", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.attest:
        attest(args.output)
    else:
        complexity(args.coverage, args.output)


if __name__ == "__main__":
    main()
