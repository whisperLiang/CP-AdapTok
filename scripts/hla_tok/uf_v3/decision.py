"""Freeze the capacity diagnosis from the audited Train holdout."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.v2.protocol import ROOT, json_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results
    audit, capacity = out / "audit/training_audit.json", out / "diagnostics/calibration/capacity.json"
    audit_data, capacity_data = json_at(audit), json_at(capacity)
    if audit_data["end_to_end"] or audit_data["test_pixel_reads"] or capacity_data["split"] != "calibration":
        raise ValueError("R0 requires the bounded V2 training finding and a Train-only capacity experiment")
    evidence = {}
    for seed, candidate in capacity_data["results"].items():
        q = candidate["pooled"]
        continuous, full_d0, full_d1, d1_40 = (q[name] for name in (
            "C0/CONTINUOUS", "C2/ALL_VALID_RECORDS", "C3/ALL_VALID_RECORDS", "C3/40"))
        evidence[seed] = {
            "c0_minus_c2_db": continuous["global_psnr"] - full_d0["global_psnr"],
            "c0_minus_c2_lpips": continuous["lpips"] - full_d0["lpips"],
            "c2_minus_c3_db": full_d0["global_psnr"] - full_d1["global_psnr"],
            "c2_minus_c3_lpips": full_d0["lpips"] - full_d1["lpips"],
            "c3_all_minus_40_db": full_d1["global_psnr"] - d1_40["global_psnr"],
            "c3_all_minus_40_lpips": full_d1["lpips"] - d1_40["lpips"],
        }
    frozen = json_at(ROOT / "results/hla_tok/uf_v2/final_verdict.json")["frozen_thresholds"]
    write_json(out / "diagnostics/capacity.json", {
        "selected_route": "R0", "architecture": "one shared original latent, existing D1 and official Base recurrence",
        "observed": evidence, "native_link": json_at(ROOT / "configs/hla_tok/uf/upstream.json")["commit"],
        "audit_sha256": sha256_file(audit), "calibration_sha256": sha256_file(capacity),
        "test_pixel_reads": 0, "frozen_quality_and_realtime": frozen,
        "controls": {"R1": "reassess maximum continuous capacity after admitted joint training",
                     "R2": "continuous D0 gains near 0.5 dB but less than 0.01 LPIPS",
                     "R3": "D0 has small PSNR advantage and large perceptual deficit on identical symbols",
                     "R4": "adding all remaining legal records has small effect on D1"},
        "validation_scope": "predefined 174-frame paired confirmation, no deployment parameter tuning"})


if __name__ == "__main__":
    main()
