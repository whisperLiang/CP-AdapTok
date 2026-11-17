"""Local crapper/dryer equivalents using measured coverage, Radon and AST tokens."""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from cp_adaptok.runtime import write_json
from scripts.quality_audit import analyze_file


def duplicates(functions):
    pairs = []
    for index, first in enumerate(functions):
        for second in functions[index + 1 :]:
            matcher = SequenceMatcher(None, first[2], second[2], autojunk=False)
            if matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8:
                pairs.append({"first": first[:2], "second": second[:2], "similarity": matcher.ratio()})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="results/fixed2048/coverage.json")
    parser.add_argument("--output", default="results/fixed2048/quality_audit.json")
    args = parser.parse_args()
    coverage = json.loads(Path(args.coverage).read_text())
    rows, functions = [], []
    for file in sorted(Path("experiments/fixed2048").glob("*.py")):
        if file.name in ("quality.py", "mutation_check.py"):
            continue
        measured, parsed = analyze_file(file.resolve(), coverage)
        rows.extend(measured)
        functions.extend(parsed)
    result = {
        "tools": "Radon + coverage + AST; named crapper/dryer unavailable",
        "functions": rows,
        "crap_at_least_10": [r for r in rows if r["crap"] >= 10],
        "duplicates_at_least_08": duplicates(functions),
    }
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "functions"}, indent=2))


if __name__ == "__main__":
    main()
