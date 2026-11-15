"""Local equivalents for unavailable crapper/dryer: coverage CRAP and AST similarity.

This reports unverified functions honestly, including the unavailable native CUDA
path. Short boundary adapters are excluded from the duplication heuristic.
"""

import ast
import json
from difflib import SequenceMatcher
from pathlib import Path

from radon.complexity import cc_visit

ROOT = Path(__file__).resolve().parents[1]


def blocks(block):
    return block.methods if hasattr(block, "methods") else [block]


def analyze_file(file, coverage):
    source = file.read_text()
    key = str(file.relative_to(ROOT))
    data = coverage.get("files", {}).get(key, {})
    executed, missing = set(data.get("executed_lines", [])), set(data.get("missing_lines", []))
    rows, functions = [], []
    for group in cc_visit(source):
        for block in blocks(group):
            lines = set(range(block.lineno, block.endline + 1))
            statements = lines & (executed | missing)
            fraction = len(statements & executed) / len(statements) if statements else 0
            crap = block.complexity**2 * (1 - fraction) ** 3 + block.complexity
            rows.append(
                {
                    "file": key,
                    "function": block.name,
                    "line": block.lineno,
                    "complexity": block.complexity,
                    "coverage": fraction,
                    "crap": crap,
                }
            )
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef):
            body = ast.Module(body=node.body, type_ignores=[])
            tokens = [
                type(n).__name__ + ":" + str(getattr(n, "attr", getattr(n, "id", ""))) for n in ast.walk(body)
            ]
            if node.end_lineno - node.lineno >= 7 and len(tokens) > 50:
                functions.append((key, node.name, tokens))
    return rows, functions


def main():
    coverage = json.loads((ROOT / "results/coverage.json").read_text())
    rows, functions = [], []
    for file in sorted((ROOT / "cp_adaptok").rglob("*.py")):
        if "tests" in file.parts:
            continue
        file_rows, file_functions = analyze_file(file, coverage)
        rows.extend(file_rows)
        functions.extend(file_functions)
    duplicates = []
    rows = list({(r["file"], r["function"], r["line"]): r for r in rows}.values())
    for i, a in enumerate(functions):
        for b in functions[i + 1 :]:
            matcher = SequenceMatcher(None, a[2], b[2], autojunk=False)
            if matcher.real_quick_ratio() < 0.8 or matcher.quick_ratio() < 0.8:
                continue
            similarity = matcher.ratio()
            if similarity >= 0.8:
                duplicates.append({"a": a[:2], "b": b[:2], "similarity": similarity})
    report = {
        "tools": "radon + coverage + AST comparison; not the unavailable named tools",
        "max_complexity": max(r["complexity"] for r in rows),
        "crap_over_10": [r for r in rows if r["crap"] >= 10],
        "duplicates_over_08": duplicates,
        "functions": rows,
    }
    (ROOT / "results/quality_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "functions"}, indent=2))


if __name__ == "__main__":
    main()
