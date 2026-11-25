"""Frozen multi-metric candidate ordering and admission rules."""

from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.inputs import OUTPUT
from scripts.hla_tok.continuous.inputs import METRICS, run_arguments
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3.protocol import read_config


def signed_quality(row: dict) -> tuple[float, ...]:
    return tuple(float(row[m]) * (1 if m == "global_psnr" else -1) for m in METRICS)


def dominates(left: tuple, right: tuple) -> bool:
    pairs = list(zip(left, right, strict=True))
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def _vectors(assets: list[str], index: dict, caps: list[int]) -> dict[str, tuple]:
    return {asset: tuple(v for cap in caps for v in signed_quality(index[asset, cap])) for asset in assets}


def _ranks(values: dict[str, tuple]) -> dict[str, int]:
    return {key: sum(a > b for vector in values.values() for a, b in zip(vector, scores, strict=True))
            for key, scores in values.items()}


def ranked_profiles(rows: list[dict], config: dict) -> list[dict]:
    primary = config["quantization"]["comparison_primary_order"]
    caps = config["frame_wire_byte_caps"]
    rows = [r for r in rows if r["order"] == primary]
    index = {(row["asset"], int(row["cap_bytes"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate profile bandwidth observation")
    profiles = {row["asset"]: row["profile"] for row in rows}
    assets = [asset for asset in profiles if all(float(index[asset, cap]["overflow_frame_fraction"]) == 0
                                               and int(index[asset, cap]["frames"]) == 174 for cap in caps)]
    if not assets:
        raise ValueError("no complete byte-feasible profiles")
    low = config["selection"]["low_rate_caps"]
    vlow = _vectors(assets, index, low)
    vhigh = _vectors(assets, index, [cap for cap in caps if cap not in low])
    low_ranks = _ranks(vlow)
    high_ranks = _ranks(vhigh)
    priority = config["quantization"]["profiles"]
    records = [{"asset": asset, "profile": profiles[asset],
                "dominated_low_rate": any(dominates(values, vlow[asset]) for values in vlow.values()),
                "low_rate_rank_sum": low_ranks[asset], "high_rate_rank_sum": high_ranks[asset]}
               for asset in assets]
    return sorted(records, key=lambda r: (r["dominated_low_rate"], r["low_rate_rank_sum"], r["high_rate_rank_sum"],
                                          priority.index(r["profile"])))


def stage_targets(bands: list[dict], equal: dict, config: dict) -> dict:
    limits = config["equal_payload"]
    psnr, lpips, dists = map(float, (equal["global_psnr"], equal["lpips"], equal["dists"]))
    equal_gate = psnr >= limits["min_psnr"] and lpips <= limits["max_lpips"] and dists <= limits["max_dists"]
    gain = psnr - limits["min_psnr"]
    material_perceptual = limits["material_perceptual"]
    material = equal_gate and (gain >= limits["material_db"] or
                               (limits["max_lpips"] - lpips >= material_perceptual["lpips"]
                                and limits["max_dists"] - dists >= material_perceptual["dists"]))
    rates = {float(r["target_mbps"]): r for r in bands}
    practical = {str(band): {"pass": float(rates[band]["global_psnr"]) >= target,
                              "psnr_db": float(rates[band]["global_psnr"]), "target_db": target}
                 for band, target in config["stage_psnr_targets"].items()}
    return {"equal_payload_pass": equal_gate, "equal_payload_material": material, "equal_payload_gain_db": gain,
            "practical": practical, "all_practical_pass": all(row["pass"] for row in practical.values())}


def _progressive_cases(summary: list[dict], config: dict, asset: str) -> list[dict]:
    order = config["quantization"]["comparison_primary_order"]
    lookup = {int(r["budget"]): r for r in summary if r["asset"] == asset and r["order"] == order}
    tolerance = config["projection"]["tolerances"]
    rows = []
    for a, b in zip(config["budgets"], config["budgets"][1:]):
        delta = dict(zip(METRICS, (b - a for a, b in zip(signed_quality(lookup[a]), signed_quality(lookup[b]),
                                                      strict=True)), strict=True))
        rows.append({"from_k": a, "to_k": b, **{f"gain_{m}": d for m, d in delta.items()},
                     "pass": all(value >= -tolerance[key] for key, value in delta.items())})
    return rows


def select(output: Path, config: dict) -> dict:
    for filename in ("prefix_audit.json", "wire_audit.json", "causality_audit.json"):
        if json_at(output / filename)["status"] != "PASS":
            raise ValueError(f"real native correctness gate failed: {filename}")
    rows = csv_at(output / "bandwidth_constrained_quality.csv")
    ranked = ranked_profiles(rows, config)
    primary = config["quantization"]["comparison_primary_order"]
    best = ranked[0]["asset"]
    bands = [r for r in rows if r["asset"] == best and r["order"] == primary]
    equal, = [r for r in csv_at(output / "compact_equal_caps.csv")
              if r["asset"] == best and r["order"] == primary and r["basis"] == "payload_only"]
    targets = stage_targets(bands, equal, config)
    progressive = _progressive_cases(csv_at(output / "scalar_quantization_rd.csv"), config, best)
    write_csv(output / "progressive_reconstruction.csv", progressive)
    result = {"main_asset": best, "order": primary, "ranking": ranked, "targets": targets,
              "first_saturated_band_mbps": next((float(r["target_mbps"]) for r in bands if int(r["budget"]) == MAX_K), None),
              "progressive": "PASS" if all(row["pass"] for row in progressive) else "PARTIAL PASS"}
    write_json(output / "no_training_selection.json", result)
    return result


def main() -> None:
    from scripts.hla_tok.compact.inputs import CONFIG

    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    select(args.output, read_config(CONFIG))


if __name__ == "__main__":
    main()
