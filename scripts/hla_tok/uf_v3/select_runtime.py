"""Select the byte-identical workspace build after physical runtime and contract replay."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.uf_v3.evaluate import close_timing
from scripts.hla_tok.v2.protocol import json_at


def select(root, binary):
    work = root / "runtime_trial"
    paths = [work / f"validation_{name}/summary.json" for name in ("full_max", "on_demand")]
    attempts = [json_at(path) for path in paths]
    a, b = (json_at(work / f"validation_{name}_family.json") for name in ("original", "reuse"))
    parity = a == b and len(a["frames"]) == 174
    closure = json_at(work / "reuse_contracts/summary.json")
    proven = closure["status"] == "PASS" and closure["deployment"]["status"] == "PASS"
    checkpoint = closure["deployment"]["checkpoint_sha256"]
    clocks = all(row["checkpoint_sha256"] == checkpoint and row["split"] == "validation"
                 and row["shape"] == [1, 3, 1080, 1920] and row["timing_mode"] == mode
                 and row["other_gpu_compute_before_after"] == [0, 0] and close_timing(row["timing"]) == "PASS"
                 for row, mode in zip(attempts, ("Full-max", "on-demand"), strict=True))
    return {"selected": "reuse" if parity and proven and clocks else "original",
            "max_and_all_boundary_hashes_match": parity,
            "native_frames_with_paired_hashes": len(a["frames"]), "contract_replay": proven,
            "both_modes_realtime": clocks, "native_record_extension_sha256": sha256_file(binary),
            "timing": {key: str(path.relative_to(root)) for key, path in zip(
                ("full_max", "on_demand"), paths, strict=True)},
            "contracts": "runtime_trial/reuse_contracts/summary.json", "test_pixel_reads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    write_json(args.results / "runtime_trial/selection.json", select(args.results, args.binary))


if __name__ == "__main__":
    main()
