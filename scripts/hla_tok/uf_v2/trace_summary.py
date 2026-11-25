"""Summarize observed CUDA activities without adding overlapping durations."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from hla_tok.evaluation.io import write_json


def union_duration(events):
    end, total = 0, 0.0
    for start, duration in sorted((row["ts"], row["dur"]) for row in events):
        stop = start + duration
        total += max(0, stop - max(end, start))
        end = max(end, stop)
    return total


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    by_category = defaultdict(list)
    for event in events:
        by_category[event.get("cat")].append(event)
    kernels = by_category["kernel"]
    transfers = defaultdict(lambda: {"count": 0, "bytes": 0})
    for event in by_category["gpu_memcpy"]:
        row = transfers[event["name"]]
        row["count"] += 1
        row["bytes"] += event["args"].get("bytes", 0)
    sync = [row for cat in ("cuda_driver", "cuda_runtime") for row in by_category[cat]
            if "Synchronize" in row["name"]]
    allocations = [row["args"]["Bytes"] for row in events
                   if row.get("name") == "[memory]" and row["args"].get("Bytes", 0) > 0]
    gpu_work = kernels + by_category["gpu_memcpy"] + by_category["gpu_memset"]
    window = [row for row in by_category["user_annotation"] if row["name"] == "application.frame"]
    return {
        "scope": "kineto observed events only; the pinned native proxy CUDA graph may be opaque",
        "kernels": len(kernels), "kernel_sum_ms": sum(row["dur"] for row in kernels) / 1000,
        "kernel_union_ms": union_duration(kernels) / 1000,
        "gpu_activity_union_ms": union_duration(gpu_work) / 1000,
        "transfers": transfers, "observed_sync_count": len(sync),
        "torch_alloc_count": len(allocations), "torch_alloc_bytes": sum(allocations),
        "frame_window_ms": union_duration(window) / 1000,
        "frames": len(window),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, {path.name: summarize(path) for path in sorted(args.traces.glob("*.trace.json"))})


if __name__ == "__main__":
    main()
