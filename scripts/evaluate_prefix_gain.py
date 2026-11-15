"""Summarize actual stream RD and sequence-bootstrap gains without extrapolation."""

import argparse
import csv
import json

import numpy as np
from scipy.interpolate import PchipInterpolator

from cp_adaptok.evaluation import budget_summary, frame_type_summaries, prefix_gain_summary
from cp_adaptok.runtime import resolve, write_json


def rd_curve(rows):
    quality = np.asarray([row["psnr"] for row in rows])
    rates = np.asarray([row["bpp"] for row in rows])
    if len(rows) < 4 or np.any(np.diff(quality) <= 0) or np.any(np.diff(rates) <= 0):
        return None
    return quality, np.log(rates)


def bd_rate(reference, candidate):
    curves = [rd_curve(rows) for rows in (reference, candidate)]
    if any(curve is None for curve in curves):
        return {"percent": None, "reason": "need >=4 strictly increasing measured quality/rate points"}
    lower = max(x[0][0] for x in curves)
    upper = min(x[0][-1] for x in curves)
    if lower >= upper:
        return {"percent": None, "reason": "measured quality ranges do not overlap; no extrapolation"}
    integrals = [PchipInterpolator(x, y, extrapolate=False).integrate(lower, upper) for x, y in curves]
    return {"percent": float(100 * np.expm1((integrals[1] - integrals[0]) / (upper - lower))),
            "overlap_psnr": [float(lower), float(upper)], "method": "PCHIP log-rate, shared measured quality only"}


def report_summary(report, path):
    if not report.get("real_entropy_bitstream"):
        raise ValueError(f"RD requires actual bitstream measurements: {path}")
    records = report["records"]
    budgets = [p["tokens"] for p in records[0]["prefixes"]]
    return {
        "source": str(resolve(path)), "model_id": report["model_id"],
        "absolute": budget_summary(records, budgets), "gains": prefix_gain_summary(records),
        "by_frame_type": frame_type_summaries(records, budgets),
        "metric_errors": report["metric_errors"], "frames": len(records),
    }


def flat_report_rows(label, summary):
    cases = {"all": summary["absolute"], **summary["by_frame_type"]}
    return [dict(model=label, frame_type=kind, **point) for kind, rows in cases.items() for point in rows]


def summarize_reports(paths, labels, output):
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, rows = {}, []
    for label, path in zip(labels, paths):
        report = json.loads(resolve(path).read_text())
        summaries[label] = report_summary(report, path)
        rows.extend(flat_report_rows(label, summaries[label]))
    reference = summaries[labels[0]]["absolute"]
    comparisons = {label: bd_rate(reference, summaries[label]["absolute"]) for label in labels[1:]}
    write_json(output / "summary.json", {"models": summaries, "bd_rate_vs_first": comparisons})
    with (output / "rd_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_rd(summaries, output)


def plot_rd(summaries, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, summary in summaries.items():
        rows = summary["absolute"]
        for ax, metric in zip(axes, ("psnr", "lpips", "dists")):
            if all(row[metric] is not None for row in rows):
                ax.plot([row["bpp"] for row in rows], [row[metric] for row in rows], "o-", label=label)
            ax.set(xlabel="Actual bpp (application framing included)", ylabel=metric)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "rd_curves.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="results/prefix_utilization/full_1080p/comparison")
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        raise ValueError("provide exactly one distinct label for each measured report")
    summarize_reports(args.inputs, args.labels, args.output)


if __name__ == "__main__":
    main()
