"""Measured pilot figures: one seed, no synthetic points or uncertainty intervals.

Figure contract: test whether the matched VQ changes restore Fine information and
improve actual-prefix rate distortion. Single-panel quantitative plots separate
training dynamics from same-protocol Validation comparisons. All source rows are
retained in source_data.json; smoothing is declared, never used for selection.
Export: editable SVG/PDF plus 600-DPI PNG, 7+ pt text, and rendered PDF QA.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptok_hr.runtime import write_json

ROOT = Path("results/adaptok_hr_fine_recovery")
COLORS = {"A0": "#777777", "A1": "#8467A8", "A2": "#398B8D", "A3": "#CF8246", "baseline": "#222222"}
STYLES = {"A0": "--", "A1": ":", "A2": "-.", "A3": "-", "baseline": "--"}


def canvas(ylabel, xlabel="Optimizer updates"):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(direction="out")
    return fig, ax


def export(fig, ax, name, output, alignment):
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.21), ncol=5, frameon=False)
    fig.subplots_adjust(left=.14, right=.97, bottom=.18, top=.81)
    fig.canvas.draw()
    alignment(fig, json_out=str(output / f"{name}.alignment.json"), strict=True)
    fig.savefig(output / f"{name}.svg")
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=600)
    plt.close(fig)


def training_curves(output, alignment):
    source = {}
    figures = {key: canvas(label) for key, label in (
        ("training_psnr", "Training PSNR (25-update pooled MSE, dB)"),
        ("fine_usage", "Fine used IDs (mean over active Fine frames)"),
        ("fine_max_fraction", "Fine maximum ID fraction (active frames)"),
        ("validation_psnr", "K2048 short Validation global PSNR (dB)"))}
    for name in ("A0", "A1", "A2", "A3"):
        rows = [json.loads(line) for line in (ROOT / name / "training.jsonl").read_text().splitlines()]
        source[name] = rows
        x = np.array([r["step"] for r in rows])
        mse = np.array([r["mse"] for r in rows])
        smooth = np.convolve(mse, np.ones(25) / 25, mode="valid")
        style = {"label": name, "color": COLORS[name], "linestyle": STYLES[name], "linewidth": 1.2}
        figures["training_psnr"][1].plot(x[24:], -10 * np.log10(np.maximum(smooth, 1e-20)), **style)
        for key, metric in (("fine_usage", "used_codes"), ("fine_max_fraction", "maximum_code_fraction")):
            values = []
            for row in rows:
                fine = [f["levels"]["Fine"][metric] for f in row["frame_metrics"] if "Fine" in f["levels"]]
                values.append(float(np.mean(fine)) if fine else np.nan)
            figures[key][1].plot(x, values, **style)
        checkpoints = sorted((ROOT / name).glob("validation_*.json"))
        xx = [int(p.stem.split("_")[1]) for p in checkpoints]
        yy = [json.loads(p.read_text())["budgets"]["2048"]["psnr_global"] for p in checkpoints]
        figures["validation_psnr"][1].plot(xx, yy, marker="o", markersize=3, **style)
    for name, (fig, ax) in figures.items():
        export(fig, ax, name, output, alignment)
    return source


def rate_distortion(output, alignment):
    source = {}
    for steps, frames in ((300, 8), (1000, 64)):
        paths = sorted(ROOT.glob(f"*/validation_{steps:06d}_{frames}f/summary.json"))
        if not paths:
            continue
        for metric, ylabel in (("psnr_global", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex (lower is better)"),
                               ("dists", "DISTS (lower is better)")):
            fig, ax = canvas(ylabel, "Actual bitstream rate (bits / pixel)")
            for path in paths:
                name = path.parents[1].name
                result = json.loads(path.read_text())
                source[str(path)] = result
                values = [result["closed_loop"][str(k)] for k in (256, 512, 1024, 2048)]
                ax.plot([v["actual_bpp"] for v in values], [v[metric] for v in values],
                        label=name, color=COLORS.get(name, "#222222"), linestyle=STYLES.get(name, "--"),
                        marker="o", markersize=3, linewidth=1.2)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            export(fig, ax, f"rd_{steps}_{frames}f_{metric}", output, alignment)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                               "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                               "ytick.labelsize": 7, "legend.fontsize": 8,
                               "pdf.fonttype": 42, "svg.fonttype": "none"})
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source = {"training": training_curves(output, require_matplotlib_panel_alignment),
              "validation": rate_distortion(output, require_matplotlib_panel_alignment)}
    write_json(output / "source_data.json", source)
    (output / "figure_contract.md").write_text(
        "One shared initialization seed, no independent-seed uncertainty estimate. All four configurations have identical data/budget plans. "
        "Training MSE uses a trailing 25-update pooled mean; Fine usage/fraction average only frames with active Fine, absent observations stay as gaps. "
        "Rate-distortion panels use actual bytes and independent budget loops, one point for each of K256/512/1024/2048. "
        "300-update curves use 24 Validation frames; 1000-update curves use the fixed 174-frame subset. Never compare these as matched sample sizes. "
        "Single-panel alignment is not applicable. No error bars: one seed, correlated video frames. Source JSON preserves every row.\n")


if __name__ == "__main__":
    main()
