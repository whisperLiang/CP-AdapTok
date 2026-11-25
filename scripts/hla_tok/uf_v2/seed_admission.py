"""Check the frozen Seed 2 formal-training admission from Seed 1 measurements."""

import argparse
import json
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.uf_v2.acceptance import native_gates, timing_met
from scripts.hla_tok.uf_v2.continuation import low_rd, physical_summary
from scripts.hla_tok.v2.protocol import ROOT


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def realtime(measured, cutoffs):
    if measured["size_hw"] != [1080, 1920] or measured["mode"] != "full-max":
        raise ValueError("Seed 2 admission requires native 1080p full-max timing")
    return {key: timing_met(measured["timing"][key], cutoffs) for key in ("5", "10", "20", "40")}


def decide(root, runtime_path, closure_path, incremental_path, protocol):
    trace = read(root / "training/formal-seed1/continuation.json")
    terminal = root / "d1/formal-seed1-step20000-validation/summary.json"
    completed = (trace["status"] == "CONTINUE" and terminal.is_file()) or (
        trace["status"] == "STOP_TWO_CONSECUTIVE_WITHOUT_GAIN")
    if not completed:
        return {"status": "NOT_ADMITTED", "reason": trace["status"], "terminal_20000": terminal.is_file()}
    start = physical_summary(Path(trace["records"][0]["path"]))
    best = Path(min(trace["records"], key=lambda row: row["low_rd"])["path"])
    final = physical_summary(best)
    improved = {str(b): low_rd(final[b]) < low_rd(start[b]) for b in (5, 10)}
    measured, closure, incremental = map(read, (runtime_path, closure_path, incremental_path))
    if measured["checkpoint_sha256"] != read(best)["checkpoint_sha256"]:
        raise ValueError("runtime evidence belongs to a different checkpoint")
    physical_summary(runtime_path)
    runtime = realtime(measured, protocol["deployment_thresholds"])
    native = native_gates(closure, incremental, measured["model_tag"], protocol["base"]["bytes_max"])
    admitted = all(native.values()) and all(improved.values()) and all(runtime.values())
    return {"status": "ADMITTED" if admitted else "NOT_ADMITTED", "gain_by_low_cap": improved,
            "realtime_by_cap": runtime, "native_gates": native, "best_e0_summary": str(best),
            "runtime_summary": str(runtime_path), "closure_summary": str(closure_path),
            "incremental_summary": str(incremental_path), "stream_model_tag": measured["model_tag"],
            "criterion": "both low-cap composites improve, all four real-time caps pass, causal closure and append pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument("--closure-summary", type=Path, required=True)
    parser.add_argument("--incremental-summary", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(ROOT / "configs/hla_tok/uf_v2/pilot.json")
    result = decide(args.results, args.runtime_summary, args.closure_summary, args.incremental_summary, protocol)
    write_json(args.results / "training/seed2_admission.json", result)
    print(result["status"])


if __name__ == "__main__":
    main()
