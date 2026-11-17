"""Repository's radon/coverage/AST equivalents for unavailable crapper/dryer tools."""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from scripts.quality_audit import analyze_file

from .artifacts import write_json
from .model_adapter import ROOT


def duplicates(functions):
    result = []
    for index, a in enumerate(functions):
        for b in functions[index + 1:]:
            matcher = SequenceMatcher(None, a[2], b[2], autojunk=False)
            if matcher.real_quick_ratio() >= 0.8 and matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8:
                result.append({"a": a[:2], "b": b[:2], "similarity": matcher.ratio()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=ROOT / "results/original_adaptok_1080p/coverage.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/original_adaptok_1080p/quality_audit.json")
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text())
    rows, functions = [], []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        file_rows, file_functions = analyze_file(path, coverage)
        rows.extend(file_rows)
        functions.extend(file_functions)
    report = {"tools": "radon + statement coverage + AST SequenceMatcher; named crapper/dryer unavailable",
              "functions": rows, "crap_over_10": [r for r in rows if r["crap"] >= 10], "duplicates_over_08": duplicates(functions)}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "functions"}, indent=2))
    if report["crap_over_10"] or report["duplicates_over_08"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
