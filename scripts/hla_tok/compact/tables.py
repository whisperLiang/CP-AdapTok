"""Join verified full-cohort quality and rate ledgers at common caps."""

from pathlib import Path

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.decision import dominates, signed_quality
from scripts.hla_tok.compact.inputs import CONFIG
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3.protocol import read_config

METRICS = ("global_psnr", "lpips", "dists")
EQUAL_BITS = (("Q32-4", "Q16-8"), ("Q16-4", "Q8-8"), ("Q8-4", "Q4-8"))


def _wire_point(rows: list[dict], cap: int) -> dict | None:
    return max((row for row in rows if float(row["total_bytes_per_frame"]) <= cap),
               key=lambda row: float(row["total_bytes_per_frame"]), default=None)


def _best_available(rows: list[dict], priorities: list[str]) -> dict:
    quality = [signed_quality(r) for r in rows]
    indexed = list(zip(rows, quality, strict=True))

    def priority(item):
        row, values = item
        return (any(dominates(other, values) for other in quality),
                sum(a > b for other in quality for a, b in zip(other, values, strict=True)),
                priorities.index(row["profile"]))

    return min(indexed, key=priority)[0]


def _comparison_columns(label: str, row: dict | None) -> dict:
    if row is None:
        return {label + "_status": "NOT_COMPARABLE"}
    return {label + "_status": "MEASURED", **{f"{label}_{m}": float(row[m]) for m in METRICS},
            **{f"{label}_{k}": row.get(k, "") for k in ("profile", "budget", "total_bytes_per_frame", "mbps_30fps")}}


def bandwidth_comparison(output: Path, config: dict, selection: dict, training: dict) -> list[dict]:
    legacy = csv_at(output / "single_frame_references.csv")
    compact = csv_at(output / "bandwidth_constrained_quality.csv")
    original = csv_at(output / "original_rd.csv")
    bands = zip(config["bandwidth_mbps"], config["frame_wire_byte_caps"], strict=True)
    order = selection["order"]
    sources = [r for r in compact if r["order"] == order]
    priorities = config["quantization"]["profiles"]
    c1, = [r for r in legacy if r["method"] == "C1 Fixed13"]
    if training.get("adopted"):
        trained = csv_at(output / "training_final_bandwidth.csv")
        sources.extend(r | {"cap_bytes": r["cap"]} for r in trained if r["basis"] == "total_wire")
    main = training["asset"] if training.get("adopted") else selection["main_asset"]
    comparison = []
    for band, cap in bands:
        c0, = [r for r in legacy if r["method"] == "C0 BF16" and int(r["cap_bytes"]) == cap]
        rows = [r for r in sources if int(r["cap_bytes"]) == cap]
        best = _best_available(rows, priorities)
        fixed, = [r for r in rows if r["asset"] == main]
        a = _wire_point([c1], cap)
        b = _wire_point(original, cap)
        gain = signed_quality(best)[0] - signed_quality(c0)[0]
        comparison.append({
            "target_mbps": band, "cap_bytes": cap, "best_compact_asset": best["asset"], "main_asset": main,
            "gain_vs_c0_db": gain, "original_time_basis": "16-frame window average" if b else "",
            **_comparison_columns("c0", c0), **_comparison_columns("c1_f13w", a),
            **_comparison_columns("original", b), **_comparison_columns("best_compact", best),
            **_comparison_columns("main_compact", fixed),
        })
    return comparison


def equal_rate(output: Path, config: dict) -> list[dict]:
    legacy = csv_at(output / "single_frame_references.csv")
    c1, = [row for row in legacy if row["method"] == "C1 Fixed13"]
    archival_cap = config["equal_payload"]["cap_bytes"]
    fixed_wire_cap = int(float(c1["total_bytes_per_frame"]))
    baseline = [{
        "method": "C1 Fixed13 archive", "basis": "equal_payload_cap", "cap_bytes": archival_cap,
        "comparability": "PASS", "budget": MAX_K, "payload_bytes_per_frame": archival_cap,
        **dict(zip(METRICS, (config["equal_payload"]["min_psnr"], config["equal_payload"]["max_lpips"],
                            config["equal_payload"]["max_dists"]), strict=True)),
    }, {
        "method": "C1 Fixed13 archive", "basis": "equal_total_wire_cap", "cap_bytes": archival_cap,
        "comparability": "NOT_COMPARABLE", "reason": "archive supplies physical payload without a full header ledger",
    }, c1 | {"method": "C1 Fixed13 F13W", "basis": "equal_total_wire_cap", "cap_bytes": fixed_wire_cap,
             "comparability": "PASS", "reason": "independent evaluation envelope"}]
    for r in legacy:
        if r["method"] == "C0 BF16" and int(r["cap_bytes"]) == -archival_cap:
            baseline.append(r | {"basis": "equal_payload_cap", "cap_bytes": archival_cap, "comparability": "PASS"})
    for row in csv_at(output / "compact_equal_caps.csv"):
        mode = "equal_payload_cap" if row["basis"] == "payload_only" else "equal_total_wire_cap"
        cap = int(row["cap_bytes"])
        comparability = "PASS" if mode == "equal_payload_cap" or cap == fixed_wire_cap else "NOT_COMPARABLE"
        signed = [a - b for a, b in zip(signed_quality(row), signed_quality(c1), strict=True)]
        baseline.append(row | {"method": "Compact", "basis": mode, "comparability": comparability,
                               **{f"gain_vs_c1_{m}": gain for m, gain in zip(METRICS, signed, strict=True)}})
    return baseline


def ablation(output: Path, config: dict) -> None:
    order = config["quantization"]["comparison_primary_order"]
    budgets = config["budgets"]
    scalar = csv_at(output / "scalar_quantization_rd.csv")
    projection = csv_at(output / "projection_only_rd.csv")
    idx = {(r["profile"], r["projection"], int(r["budget"])): r for r in projection if r["order"] == order}
    q = {(r["profile"], int(r["budget"])): r for r in scalar if r["order"] == order and r["profile"] != "RAW32"}
    records, equal_bits = [], []
    for (label, k), row in q.items():
        if k not in budgets:
            continue
        spec = PROFILES_BY_LABEL[label]
        parent = "RAW32" if spec.dimension == 32 else f"PCA{spec.dimension}-BF16"
        base = idx[parent, row["projection"], k]
        delta = [a - b for a, b in zip(signed_quality(row), signed_quality(base), strict=True)]
        records.append({"profile": label, "projection": row["projection"], "budget": k, "bypass": parent,
                        "quantized_total_bpp": float(row["total_bpp"]), "bypass_total_bpp": float(base["total_bpp"]),
                        **{f"quantized_gain_{m}": gain for m, gain in zip(METRICS, delta, strict=True)}})
    for a, b in EQUAL_BITS:
        for k in budgets:
            first, second = q[a, k], q[b, k]
            if first["total_bytes"] != second["total_bytes"]:
                raise ValueError("equal-bit ablation observed different physical wire rates")
            gains = [x - y for x, y in zip(signed_quality(first), signed_quality(second), strict=True)]
            equal_bits.append({"higher_dimension": a, "lower_dimension": b, "budget": k,
                               "bytes_per_frame": first["total_bytes_per_frame"],
                               **{f"higher_dimension_gain_{m}": d for m, d in zip(METRICS, gains, strict=True)}})
    write_csv(output / "projection_quantization_ablation.csv", records)
    write_csv(output / "equal_bits_ablation.csv", equal_bits)


def _raw_parity(output: Path) -> None:
    def key(row: dict) -> tuple:
        return row["video"], row["frame"], row["order"], row["budget"]

    old = {key(r): r for r in csv_at(output / "projection_only_per_frame.csv")
           if r["profile"] == "RAW32"}
    pairs = [(r, old[key(r)]) for r in csv_at(output / "scalar_quantization_per_frame.csv")
             if r["profile"] == "RAW32" and key(r) in old]
    fields = ("mse", "psnr", "lpips", "dists")
    maximum = {name: max(abs(float(a[name]) - float(b[name])) for a, b in pairs) for name in fields}
    write_json(output / "raw_bf16_matrix_parity.json", {"matched_rows": len(pairs),
                                                       "max_absolute_difference": maximum,
                                                       "status": "PASS" if not any(maximum.values()) else "PARTIAL PASS"})


def write_tables(output: Path) -> None:
    config = read_config(CONFIG)
    selection = json_at(output / "no_training_selection.json")
    training = json_at(output / "training_selection.json")
    write_csv(output / "equal_rate_comparison.csv", equal_rate(output, config))
    write_csv(output / "bandwidth_baseline_comparison.csv", bandwidth_comparison(output, config, selection, training))
    ablation(output, config)
    _raw_parity(output)
