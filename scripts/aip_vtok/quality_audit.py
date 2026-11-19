"""CRAP/duplication equivalents with measured coverage; missing coverage stays missing."""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from scripts.quality_audit import analyze_file

ROOT = Path(__file__).resolve().parents[2]


def duplicate_pairs(functions):
    rows = []
    for index, first in enumerate(functions):
        for second in functions[index + 1 :]:
            matcher = SequenceMatcher(None, first[2], second[2], autojunk=False)
            if matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8:
                rows.append({"first": first[:2], "second": second[:2], "similarity": matcher.ratio()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="results/aip_vtok/coverage.json")
    args = parser.parse_args()
    coverage = json.loads((ROOT / args.coverage).read_text())
    rows, functions = [], []
    for path in sorted((ROOT / "aip_vtok").rglob("*.py")):
        measured, blocks = analyze_file(path, coverage)
        rows.extend(measured)
        functions.extend(blocks)
    result = {
        "tools": "coverage/Radon/AST equivalents; named crapper/dryer unavailable",
        "crap_at_least_10": [r for r in rows if r["crap"] >= 10],
        "duplicates_at_least_08": duplicate_pairs(functions),
        "functions": rows,
    }
    (ROOT / "results/aip_vtok/quality_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "functions"}, indent=2))


if __name__ == "__main__":
    main()
