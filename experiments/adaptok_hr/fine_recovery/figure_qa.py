"""Run source and rendered QA on primary figures, excluding QA overlay PDFs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from adaptok_hr.runtime import write_json


def audit(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    root = Path("results/adaptok_hr_fine_recovery/figures")
    source = audit([sys.executable, str(args.qa_scripts / "validate_figure.py"),
                    "experiments/adaptok_hr/fine_recovery/figures.py", "--json"])
    write_json(root / "source_audit.json", source)
    rows = []
    for pdf in sorted(root.glob("*.pdf")):
        if ".collisions-overlay" in pdf.name:
            continue
        commands = [("text", ["audit_pdf_text.py", str(pdf), "--min-pt", "5", "--json"]),
                    ("collisions", ["audit_figure_collisions.py", str(pdf), "--json-out",
                                     str(pdf.with_suffix(".collisions.json")), "--overlay-pdf",
                                     str(pdf.with_suffix(".collisions-overlay.pdf"))])]
        for kind, command in commands:
            record = audit([sys.executable, str(args.qa_scripts / command[0]), *command[1:]])
            rows.append({"file": str(pdf), "kind": kind, **record})
    write_json(root / "rendered_qa.json", rows)
    failures = [r for r in (source, *rows) if r["returncode"]]
    print(json.dumps({"primary_figures": len(rows) // 2, "failed_checks": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
