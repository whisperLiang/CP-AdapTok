"""Check that Phase 0 tests reject focused changes to the native contracts."""

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.uf_v2.mutations import Fault, Finder

UNITS = "hla_tok.hts.units"
COMPARISON = "scripts.hla_tok.hts.native_comparison"
FAULTS = (
    Fault("read real future during I", UNITS, "yield Unit(first, 1, 1, True, False)",
          "yield Unit(first, 1, 8, True, False)",
          "test_units.py::test_first_frame_stands_alone_and_full_units_keep_order"),
    Fault("repeat the first frame in a tail", UNITS, "unit.indices[-1]", "unit.indices[0]",
          "test_hts_inputs.py::test_native_input_reads_only_the_coded_unit_and_copies_the_tail"),
    Fault("skip the requested feature reset", UNITS, "yield Unit(start, valid, self.delay, False, reset)",
          "yield Unit(start, valid, self.delay, False, False)",
          "test_units.py::test_matched_gop_keeps_four_whole_chunks_and_explicit_i_restarts"),
    Fault("ignore I byte overage", COMPARISON, 'if row["total_bytes"] <= int(bytes_per_frame * row["valid_frames"])',
          'if row["unit_type"] == "I" or row["total_bytes"] <= int(bytes_per_frame * row["valid_frames"])',
          "test_comparison.py::test_strict_fit_charges_all_bytes_against_valid_frame_time"),
    Fault("permit equal-quality extrapolation", "scripts.hla_tok.uf_v2.rate_analysis",
          "if not points or not points[0][0] <= value <= points[-1][0]:",
          "if not points:", "test_comparison.py::test_equal_quality_comparison_requires_overlap_and_saves_physical_bytes"),
    Fault("allow Test pixel access", "hla_tok.evaluation.data", 'raise PermissionError("Test remains sealed")',
          "return {}", "test_hts_inputs.py::test_frozen_metadata_preserves_cohorts_and_seals_test_pixels"),
    Fault("round YCbCr after centering", "hla_tok.hts.native",
          "(yuv.half() - 0.5).contiguous", "(yuv - 0.5).half().contiguous",
          "test_hts_native_contracts.py::test_cuda_rgb_and_payload_match_official_script", True),
    Fault("discard the observed native DPB hash", "hla_tok.hts.native",
          "return tensor_digest(self.video.proxy.audit_reference(unit.intra or unit.reset))", 'return "0" * 64',
          "test_hts_native_contracts.py::test_cuda_rgb_and_payload_match_official_script", True),
)


def inject(case: Fault) -> int:
    import pytest

    sys.meta_path.insert(0, Finder(case))
    return int(pytest.main(["-q", "--tb=short", f"tests/hla_tok/hts/{case.test}"]))


def check(output: Path, index: int, case: Fault) -> dict:
    command = [sys.executable, "-m", __spec__.name, "--case", str(index)]
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log = output / f"case-{index}.log"
    log.write_text(run.stdout)
    killed = run.returncode == 1 and " failed" in run.stdout and "ERROR" not in run.stdout
    return asdict(case) | {"command": command, "returncode": run.returncode, "killed": killed, "log": log.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--case", type=int, choices=range(len(FAULTS)))
    args = parser.parse_args()
    if args.case is not None:
        return inject(FAULTS[args.case])
    if args.output is None:
        parser.error("--output is required for a full suite")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [check(args.output, i, case) for i, case in enumerate(FAULTS) if case.gpu == args.gpu]
    count = sum(row["killed"] for row in results)
    write_json(args.output / "summary.json", {"killed": count, "total": len(results), "cases": results})
    return int(count != len(results))


if __name__ == "__main__":
    sys.exit(main())
