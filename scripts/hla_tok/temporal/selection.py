"""Apply frozen quality gates to independently decoded application-wire results."""

from dataclasses import asdict
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.continuous.quality import summarize
from scripts.hla_tok.temporal.evaluate import MODES, experiments
from scripts.hla_tok.temporal.inputs import COMPACT, CONFIG, RESULTS
from scripts.hla_tok.v2.protocol import csv_at, json_at

BANDS = ("5", "10", "20", "40")
SIGNS = {"global_psnr": 1, "lpips": -1, "dists": -1}


def _numbers(row: dict) -> dict:
    return row | {name: float(row[name]) for name in MODES}


def _method_name(asset: str) -> str:
    if asset.startswith("T1_"):
        return "CP Q8-4 T1"
    return "CP " + asset.split("_")[0]


def _baseline(output: Path) -> list[dict]:
    baseline = json_at(output / "baseline_reproduction.json")
    rows = [row | {"method": _method_name(row["asset"]), "scope": "replayed HLCP"} for row in baseline["compact"]]
    c1 = baseline["c1_fixed13"]
    rows.append(c1 | {"method": "C1 Fixed13", "scope": "replayed F13W"})
    rows.append(c1 | {"method": "C1 Fixed13", "scope": "replayed F13W", "basis": "payload_only", "cap_bytes": 26520})
    primary = {row["metric"]: row["measured"] for row in baseline["c0"]["metrics"]}
    rows.append(primary | {"method": "C0 BF16", "scope": "replayed native maximum", "frames": 174,
                           "cap_bytes": 1044508, "total_bytes_per_frame": 1044508, "mbps_30fps": 250.68192,
                           "basis": "native_ceiling"})
    for row in csv_at(COMPACT / "original_rd.csv"):
        rows.append(_numbers(row) | {"method": "Original AdapTok", "scope": "verified archive, window average",
                                     "basis": "original_window", "cap_bytes": float(row["total_bytes_per_frame"])})
    return rows


def _ranking(matrix: list[dict]) -> tuple[str, list[dict]]:
    methods = sorted({r["method"] for r in matrix if r["predictor"] != "absolute"})
    per_rate = {(r["method"], r["bandwidth"]): r for r in matrix if r["basis"] == "total_wire"}
    valid = [m for m in methods if all((m, b) in per_rate for b in BANDS)]

    def rank(method: str, bands: tuple) -> int:
        return sum(SIGNS[name] * per_rate[method, b][name] < SIGNS[name] * per_rate[other, b][name]
                   for b in bands for name in MODES for other in valid)

    ranking = [{"method": method, "low_three_metric_rank_sum": rank(method, BANDS[:2]),
                "high_three_metric_rank_sum": rank(method, BANDS[2:])} for method in valid]
    ordered = sorted(ranking, key=lambda r: (r["low_three_metric_rank_sum"], r["high_three_metric_rank_sum"], r["method"]))
    return ordered[0]["method"], ordered


def _primary_gate(candidate: dict, baselines: list[dict], gates: dict, bandwidth: str) -> dict:
    limits = gates.get(bandwidth, {})
    strongest_psnr = max(row["global_psnr"] for row in baselines)
    best_lpips = min(row["lpips"] for row in baselines)
    best_dists = min(row["dists"] for row in baselines)
    delta_psnr = candidate["global_psnr"] - strongest_psnr
    lpips_gain = best_lpips - candidate["lpips"]
    dists_delta = candidate["dists"] - best_dists
    if bandwidth in ("5", "10"):
        quality = delta_psnr >= limits["psnr_gain"] or (
            delta_psnr >= -gates["maximum_low_rate_psnr_loss"]
            and lpips_gain >= limits["lpips_gain_at_noninferior_psnr"])
    else:
        quality = delta_psnr >= -gates["maximum_20mbps_psnr_loss"] and lpips_gain >= -gates["maximum_20mbps_lpips_loss"]
    return {"global_psnr_gain_vs_strongest": delta_psnr, "lpips_reduction_vs_strongest": lpips_gain,
            "dists_increase_vs_strongest": dists_delta,
            "baselines": sorted({row["method"] for row in baselines}),
            "pass": quality and dists_delta <= gates["maximum_dists_loss"]}


def _paired_quality(rows: list[dict], fields: tuple) -> dict:
    values: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[name] for name in fields)
        values.setdefault(key, []).append(r | {name: float(r[name]) for name in ("mse", "psnr", "lpips", "dists")})
    return {key: summarize(items) for key, items in values.items()}


def _source_checks(output: Path, method: str, matrix: list[dict], baseline: list[dict], config: dict) -> list[dict]:
    rows = csv_at(output / "validation_per_frame.csv")
    temporal = _paired_quality([r for r in rows if r["basis"] == "total_wire" and r["bandwidth"] in BANDS],
                               ("method", "bandwidth", "video"))
    cp = _paired_quality(csv_at(output / "baselines_per_frame.csv"), ("asset", "order", "budget", "video"))
    result = []
    for b in BANDS[:3]:
        cap = config["wire_caps"][b]
        available = [row for row in baseline if row["scope"] == "replayed HLCP" and row["basis"] == "total_wire"
                     and row["cap_bytes"] == cap]
        intra = [r["method"] for r in matrix if r["predictor"] == "absolute" and r["bandwidth"] == b]
        for _, _, video in sorted(key for key in temporal if key[:2] == (method, b)):
            references = [temporal[name, b, video] | {"method": name} for name in intra]
            references += [cp[row["asset"], row["order"], str(row["budget"]), video] | {"method": row["method"]}
                           for row in available]
            result.append({"video": video, "bandwidth": b} | _primary_gate(
                temporal[method, b, video], references, config["temporal_admission"], b))
    return result


def _practical_comparison(matrix: list[dict], baseline: list[dict], config: dict) -> list[dict]:
    rows = [row | {"scope": "replayed HLT native CLOSED" if row["predictor"] != "absolute" else "replayed HLT absolute"}
            for row in matrix if row["basis"] == "total_wire"]
    for band, cap in config["wire_caps"].items():
        compatible = [r for r in baseline if r["basis"] == "total_wire" and r["cap_bytes"] == cap]
        if cap >= config["equal_c1_wire_bytes"]:
            compatible.extend(r for r in baseline if r["method"] == "C1 Fixed13" and r["basis"] != "payload_only")
        original = [row for row in baseline if row["basis"] == "original_window" and row["cap_bytes"] <= cap]
        compatible.extend(sorted(original, key=lambda row: float(row["global_psnr"]), reverse=True)[:1])
        rows.extend(row | {"bandwidth": band, "cap_bytes": cap} for row in compatible)
    return rows


def run(output: Path) -> None:
    config = json_at(CONFIG)
    temporal = [_numbers(row) for row in csv_at(output / "ablation_comparison.csv")]
    baseline = _baseline(output)
    method, ranking = _ranking(temporal)
    setting = next(item.setting for item in experiments(config) if item.setting.label == method)
    source = _source_checks(output, method, temporal, baseline, config)
    gates, goals = {}, {}
    for band in BANDS:
        candidates = [r for r in temporal if r["bandwidth"] == band and r["basis"] == "total_wire"]
        selected, = [r for r in candidates if r["method"] == method]
        goals[band] = {name: SIGNS[name] * selected[name] >= SIGNS[name] * target
                       for name, target in config["target_quality"][band].items()}
        if band == "40":
            continue
        stronger = [r for r in candidates if r["predictor"] == "absolute"]
        stronger += [r for r in baseline if r["scope"] == "replayed HLCP" and r["basis"] == "total_wire"
                     and r["cap_bytes"] == config["wire_caps"][band]]
        gates[band] = _primary_gate(selected, stronger, config["temporal_admission"], band)
        gates[band]["source_passes"] = sum(item["pass"] for item in source if item["bandwidth"] == band)
    admitted = all(gates[b]["pass"] and gates[b]["source_passes"] >= config["temporal_admission"]["source_passes_required"]
                   for b in ("5", "10")) and gates["20"]["pass"]
    write_json(output / "candidate_selection.json", {"method": method, "setting": asdict(setting), "ranking": ranking,
               "quality_admission": admitted, "full_three_metric_targets": goals, "gate": gates,
               "paired_by_video": source, "candidate_scope": "four-rate temporal profiles",
               "low_rate_only_profile_probes": len({r["method"] for r in temporal}) - len(ranking) - 3,
               "tie_break": "low-rate three-metric rank sum, high-rate rank sum, method label"})
    write_csv(output / "bandwidth_comparison.csv", _practical_comparison(temporal, baseline, config))
    equal = [r for r in temporal if r["basis"] != "total_wire"]
    equal += [r for r in baseline if r["basis"] in ("payload_only", "total_wire_f13w", "native_ceiling")
              or r["method"] in ("C1 Fixed13", "Original AdapTok")]
    write_csv(output / "equal_payload_comparison.csv", equal)
    rate = csv_at(output / "validation_per_frame.csv")
    ledger = ("actual_bytes", "header_bytes", "record_header_bytes", "crc_bytes", "base_payload_bytes",
              "enhancement_payload_bytes", "motion_payload_bytes", "bitmap_bytes", "padding_bits",
              "sent_components", "visible_tokens")
    from scripts.hla_tok.continuous.matrix import groups
    write_csv(output / "rate_breakdown.csv", [labels | {name: sum(int(row[name]) for row in values) / len(values)
              for name in ledger} for labels, values in groups(rate, ("method", "bandwidth", "basis"))])
    print(f"Quality candidate {method}. Phase 2 joint admission = {admitted}.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    run(parser.parse_args().output)
