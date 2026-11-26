"""Record the exact source, runtime, checkpoint, pixel ledger, and output hashes."""

import argparse
import json
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.uf_v3.audit import BASELINE, git
from scripts.hla_tok.v2.protocol import ROOT


def evidence_files(out):
    allowed = {".json", ".csv", ".pdf", ".svg", ".tiff", ".png", ".huf", ".md", ".log", ".txt"}
    return sorted(path for path in out.rglob("*") if path.is_file() and path.suffix in allowed
                  and path.name != "experiment_manifest.json")


def reads(files, out):
    events = {}
    for file in files:
        if file.suffix != ".json":
            continue
        data = json.loads(file.read_text())
        if not isinstance(data, dict):
            continue
        counts = data.get("read_counts", data.get("reads"))
        if isinstance(counts, dict) and all(name in counts for name in ("train", "validation", "test")):
            events[str(file.relative_to(out))] = counts
        if data.get("test_pixel_reads") not in (None, 0) or isinstance(counts, dict) and counts.get("test", 0):
            raise PermissionError(f"Test ledger for {file.name} is invalid before admission")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    out = args.results
    outputs = evidence_files(out)
    sources = [path for folder in (ROOT / "hla_tok/uf_v3", ROOT / "scripts/hla_tok/uf_v3",
                                   ROOT / "tests/hla_tok/uf_v3", ROOT / "configs/hla_tok/uf_v3",
                                   ROOT / "docs/hla_tok/uf_v3")
               for path in folder.rglob("*") if path.suffix in (".py", ".sh", ".json", ".patch", ".md")]
    sources.append(ROOT / "hla_tok/uf_v2/records.cpp")
    write_json(out / "experiment_manifest.json", {
        "name": "HLA-Tok-UF-V3", "base_commit": BASELINE, "branch": git(ROOT, "branch", "--show-current"),
        "dcvc_gitlink": git(ROOT / "third_party/DCVC", "rev-parse", "HEAD"),
        "checkpoint": {"name": args.checkpoint.name, "sha256": sha256_file(args.checkpoint),
                       "bytes": args.checkpoint.stat().st_size},
        "read_ledgers": reads(outputs, out), "test_pixel_reads": 0,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(sources)},
        "result_sha256": {str(path.relative_to(out)): sha256_file(path) for path in outputs},
    })


if __name__ == "__main__":
    main()
