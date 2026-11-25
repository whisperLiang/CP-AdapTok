"""Gate adapter training using matched-budget projection headroom."""

from pathlib import Path

from hla_tok.compression.profiles import PCA_PROFILES, PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.compact.decision import signed_quality
from scripts.hla_tok.compact.matrix import Case, evaluate, summarize_matrix
from scripts.hla_tok.compact.projection import asset_key
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.v2.protocol import csv_at


def admit(inputs: Inputs, config: dict, output: Path, assets: dict, selection: dict) -> list[str]:
    targets = selection["targets"]
    if targets["equal_payload_pass"] and targets["all_practical_pass"]:
        return []
    caps = config["selection"]["low_rate_caps"]
    order = selection["order"]
    ranks = [row["asset"] for row in selection["ranking"] if assets[row["asset"]]["projection"] != "identity"]
    ranks = ranks[:config["selection"]["adaptation_profiles"]]
    rows = [r for r in csv_at(output / "bandwidth_constrained_quality.csv")
            if r["asset"] in ranks and r["order"] == order and int(r["cap_bytes"]) in caps]
    bypasses = {}
    for key in ranks:
        spec = assets[key]
        dimension = PROFILES_BY_LABEL[spec["profile"]].dimension
        bypasses[key] = asset_key(PCA_PROFILES[dimension], spec["projection"])
    cases = [Case(bypasses[r["asset"]], order, int(r["budget"])) for r in rows]
    per_frame = evaluate(inputs, config, assets, output, "adapter_projection_headroom", cases)
    totals = {(r["asset"], r["budget"]): r for r in summarize_matrix(per_frame)}
    minimum = config["selection"]["minimum_adaptation_headroom_db"]
    checks = []
    selected = set()
    for r in rows:
        key = r["asset"]
        p = totals[bypasses[key], int(r["budget"])]
        deltas = tuple(a - b for a, b in zip(signed_quality(p), signed_quality(r), strict=True))
        if deltas[0] >= minimum:
            selected.add(key)
        checks.append({"asset": key, "budget": r["budget"], "target_mbps": r["target_mbps"],
                       "bypass_asset": p["asset"], "gain_psnr_db": deltas[0],
                       "gain_lpips": deltas[1], "gain_dists": deltas[2],
                       "minimum_psnr_headroom_db": minimum})
    write_csv(output / "adapter_projection_headroom.csv", checks)
    result = [key for key in ranks if key in selected]
    write_json(output / "adapter_admission.json", {"candidates": ranks, "admitted": result, "test_pixel_reads": 0})
    return result
