"""Measure the frozen Validation source input separately from codec timing."""

import argparse
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.uf.baseline import distribution
from scripts.hla_tok.uf.inputs import inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader, plans = inputs()
    rows, opens = [], []
    checked_open = reader._open_allowed

    def timed_open(row, item):
        start = perf_counter()
        data = checked_open(row, item)
        opens.append((perf_counter() - start) * 1e3)
        return data

    with patch.object(reader, "_open_allowed", timed_open):
        for clip in plans["validation"]:
            for offset in range(clip.length):
                index = clip.start + offset
                start = perf_counter()
                reader.read(clip.split, clip.video, index)
                elapsed = (perf_counter() - start) * 1e3
                rows.append({"video": clip.video, "index": index, "audited_read_ms": opens[-1],
                             "image_decode_and_wrap_ms": elapsed - opens[-1], "total_ms": elapsed})
    write_csv(args.output / "frames.csv", rows)
    write_json(args.output / "summary.json", {
        "frames": len(rows), "read_counts": reader.counts, "network": "NOT_RUN",
        "scope": "Validation CPU input, operating system cache unspecified; audited read includes path and SHA-256",
        "statistics": {key: distribution((r[key] for r in rows), key)
                       for key in ("audited_read_ms", "image_decode_and_wrap_ms", "total_ms")},
    })


if __name__ == "__main__":
    main()
