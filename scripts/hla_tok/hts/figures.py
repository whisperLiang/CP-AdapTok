"""Export measured Calibration curves and the low-rate native peak constraint."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from scripts.hla_tok.uf_v2.rate_analysis import at_quality, frontier
from scripts.hla_tok.uf_v3.figures import Export
from scripts.hla_tok.v2.protocol import csv_at, json_at

METRICS = (("global_psnr", "Global RGB PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"))
METHODS = {
    "LD": ("#525e68", "s"),
    "HT-S": ("#16648b", "o"),
    "HT-L": ("#3b8b80", "^"),
    "Original AdapTok": ("#b66d3d", "D"),
    "V2 Seed 2 + E1": ("#747ca3", "v"),
    "V3 A4": ("#986f84", "P"),
}
LABEL = {"ld": "LD", "hts": "HT-S", "htl": "HT-L"}


def numeric(row: dict) -> dict:
    return {key: float(row[key]) for key in ("actual_mbps", *(field for field, _ in METRICS))}


def load_original(path: Path) -> list[dict]:
    return [numeric(row | {"actual_mbps": row["mbps_30fps"]}) for row in csv_at(path)]


def load_legacy(path: Path) -> list[dict]:
    return [numeric(row) for row in json_at(path)["quality"] if row["status"] == "PASS"]


def curves(points: list[dict], profile: str, controls: dict[str, list[dict]]) -> dict[str, list[dict]]:
    result = {label: [numeric(point["summary"]) for point in points
                      if point["summary"]["profile"] == profile and point["summary"]["model"] == variant]
              for variant, label in LABEL.items()}
    return result | controls


def canvas():
    fig, ax = plt.subplots(figsize=(7.20, 3.22))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.18, top=0.92)
    return fig, ax


def plot_rd(export: Export, data: dict[str, list[dict]], field: str, label: str, name: str,
            inputs: list[Path], zoom: bool = False) -> None:
    fig, ax = canvas()
    lower_is_better = field != "global_psnr"
    for method, rows in data.items():
        if not rows:
            continue
        color, marker = METHODS[method]
        sign = -1 if lower_is_better else 1
        observations, rates = zip(*frontier(rows, field), strict=True)
        if any(rate <= 0 for rate in rates):
            raise ValueError("rate plots require measured positive stream length")
        ax.plot(rates, [sign * value for value in observations], label=method, color=color,
                marker=marker, markersize=2.6, linewidth=1.45 if method == "HT-S" else 1.05)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set(xlabel="Actual rate (Mbps at 30 fps)", ylabel=label)
    if zoom:
        ax.set(xlim=(0.05, 2), xticks=[0.05, 0.1, 0.2, 0.5, 1, 2])
    fig.subplots_adjust(top=0.82)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7)
    export.write(fig, name, inputs)


def plot_equal_lpips(export: Export, data: dict[str, list[dict]], inputs: list[Path]) -> list[dict]:
    target = 0.28978
    rows = [{"method": method, "target_lpips": target, "actual_mbps": at_quality(frontier(rows, "lpips"),
                                                                               "lpips", target)}
            for method, rows in data.items()]
    observed = [row for row in rows if row["actual_mbps"] is not None]
    if not observed:
        return rows
    fig, ax = canvas()
    fig.subplots_adjust(left=0.32)
    ax.barh([row["method"] for row in observed], [row["actual_mbps"] for row in observed],
            color=[METHODS[row["method"]][0] for row in observed])
    ax.invert_yaxis()
    ax.set_xlabel("Actual Mbps at LPIPS-Alex 0.28978")
    export.write(fig, "equal_lpips", inputs)
    return rows


def peak_points(points: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for model, label in LABEL.items():
        eligible = [point for point in points if point["summary"]["model"] == model
                    and point["summary"]["profile"] == "official" and point["summary"]["actual_mbps"] <= 0.486]
        if eligible:
            chosen = min(eligible, key=lambda point: point["summary"]["lpips"])
            rows.extend(peak_units(label, chosen["units"]))
    return rows


def peak_units(label: str, rows: list[dict]) -> list[dict]:
    peaks = {"I": 0.0, "P": 0.0}
    for row in rows:
        kind = row["unit_type"]
        rate = row["total_bytes"] * 240e-6 / row["valid_frames"]
        peaks[kind] = max(peaks[kind], rate)
    return [{"method": label, "unit": kind, "peak": rate} for kind, rate in peaks.items()]


def plot_peaks(export: Export, points: list[dict], inputs: list[Path]) -> None:
    fig, ax = canvas()
    fig.subplots_adjust(left=0.24, right=0.95, top=0.82)
    rows = peak_points(points)
    for kind, marker in (("I", "D"), ("P", "o")):
        units = [row for row in rows if row["unit"] == kind]
        ax.scatter([row["peak"] for row in units], [row["method"] for row in units],
                   label="Independent I" if kind == "I" else "P unit", marker=marker, s=24)
    ax.axvline(0.486, color="#6e7883", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Peak valid-frame-normalized rate (Mbps)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, fontsize=7)
    export.write(fig, "native_ip_peaks", inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()
    native = args.study / "native_audit"
    data = {"Original AdapTok": native / "legacy/original/original_rd.csv",
            "V2 Seed 2 + E1": native / "legacy/a2/summary.json", "V3 A4": native / "legacy/a4/summary.json"}
    controls = {name: load_original(path) if path.suffix == ".csv" else load_legacy(path)
                for name, path in data.items()}
    path = native / "qp_sweep.csv"
    points = [json_at(point) for point in sorted(native.glob("*/*/point.json"))]
    inputs = [path, *data.values()]
    export = Export(args.output, args.qa)
    for profile in ("official", "matched"):
        series = curves(points, profile, controls)
        for field, label in METRICS:
            plot_rd(export, series, field, label, f"{profile}_{field}", inputs)
    native_curves = curves(points, "official", controls)
    plot_rd(export, native_curves, "lpips", "LPIPS-Alex", "low_rate", inputs, zoom=True)
    write_csv(args.output / "equal_lpips_data.csv", plot_equal_lpips(export, native_curves, inputs))
    plot_peaks(export, points, [native / "byte_ledger.jsonl", path])
    write_json(args.output / "manifest.json", {
        "data": {str(p.relative_to(native)): sha256_file(p) for p in inputs},
        "figures": export.entries, "qa": export.audits,
        "scope": "128 measured Calibration frames across 4 source clips at native 1080p",
        "frontier": "Positive measured rates on the quality-specific Pareto front. All raw QPs are retained.",
        "equal_lpips": "Log-rate interpolation inside the measured LPIPS range.",
        "peaks": "Native official endpoint with the lowest LPIPS within 0.486 Mbps average.",
        "uncertainty": "Fixed Calibration cohort and deterministic settings. No population interval estimated.",
        "legacy_scope": "Original uses sixteen-frame independent windows. V2 and V3 use independent per-frame caps.",
    })


if __name__ == "__main__":
    main()
