"""Single-panel scientific figures from observed native frames only."""

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

METRICS = {"global_psnr": "Global PSNR (dB)", "lpips": "LPIPS-Alex", "dists": "DISTS"}
COLORS = ("#156D72", "#9C533C", "#526895", "#4E5156")


def _audit(pdf, folder, scripts):
    text = subprocess.check_output([sys.executable, str(scripts / "audit_pdf_text.py"), str(pdf), "--json"], text=True)
    (folder / f"{pdf.stem}.text-audit.json").write_text(text)
    subprocess.run([sys.executable, str(scripts / "audit_figure_collisions.py"), str(pdf),
                    "--json-out", str(folder / f"{pdf.stem}.collision-audit.json")], check=True)


def _save(figure, stem, folder, scripts):
    figure.savefig(folder / f"{stem}.pdf")
    figure.savefig(folder / f"{stem}.svg")
    figure.savefig(folder / f"{stem}.png", dpi=600)
    plt.close(figure)
    _audit(folder / f"{stem}.pdf", folder, scripts)


def _curve(ax, series, field, axis):
    for index, (label, points) in enumerate(series.items()):
        marker = ("o", "s", "^", "D")[index % 4]
        ordered = sorted(points, key=lambda row: float(row[axis]))
        ax.plot([float(row[axis]) for row in ordered], [float(row[field]) for row in ordered],
                label=label, color=COLORS[index % len(COLORS)], marker=marker, markersize=3, linewidth=1.0)


def _trend(metrics, config, folder, scripts):
    series = defaultdict(list)
    full = defaultdict(list)
    for row in metrics:
        if row["stage"] == "dense":
            label = f"{row['arm']} {row['seed']} (9-frame subset)"
            (series if row["scope"] == "trend" else full)[label].append(row)
    if not series:
        return
    for field, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
        _curve(ax, series, field, "step")
        for index, points in enumerate(full.values()):
            ax.scatter([row["step"] for row in points], [row[field] for row in points],
                       facecolors="white", edgecolors="#202020", marker="D", s=21, linewidths=0.85,
                       label="Full Validation (174 frames)" if index == 0 else None, zorder=5)
        ax.axhline(config["dense_thresholds"][field], linestyle="--", color="#9B9FA5", label="Phase A threshold")
        ax.set(xlabel="Optimizer updates", ylabel=label, title="Dense reconstruction | Native 1080p")
        ax.legend(loc="best")
        _save(fig, f"dense_trend_{field}", folder, scripts)


def _rate(metrics, folder, scripts):
    series = defaultdict(list)
    for row in metrics:
        if not row.get("actual_bpp") or int(row["frames"]) != 174:
            continue
        if row["stage"] == "historical":
            if row["arm"] == "Original AdapTok":
                if row["precision"] != "fp32" or row["stride"] != "128":
                    continue
                label = "Original AdapTok, stride 128"
            elif row["arm"] in ("AIP-VTok Joint V1", "HLA-Tok R0 Fixed13"):
                label = row["arm"]
            else:
                continue
        else:
            label = f"{row['arm']} {row['seed']} @ {row['step']}"
        series[label].append(row)
    if not series:
        return
    for field in METRICS:
        fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
        _curve(ax, series, field, "actual_bpp")
        ax.set(xlabel="Actual bits per pixel", ylabel=METRICS[field], title="Native 1080p | Actual payloads")
        ax.legend(loc="best")
        _save(fig, f"rate_{field}", folder, scripts)


def figures_from(metrics, config, folder):
    mpl = matplotlib
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "figure.facecolor": "white",
    })
    default = Path.home() / ".codex/skills/nature-figure/scripts"
    scripts = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", default))
    validator = scripts / "validate_figure.py"
    if not validator.is_file():
        raise FileNotFoundError("set NATURE_FIGURE_SCRIPTS to the installed Python figure QA tools")
    validation = subprocess.check_output([sys.executable, str(validator), __file__, "--backend", "python", "--json"],
                                         text=True)
    (folder / "source-audit.json").write_text(validation)
    _trend(metrics, config, folder, scripts)
    _rate(metrics, folder, scripts)
