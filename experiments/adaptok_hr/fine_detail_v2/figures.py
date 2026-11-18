"""Show matched, measured evidence for suffix information, convergence and rate distortion.

Single quantitative panels use native packets and exact plotted source points.
Rates use actual byte counts; two repetitions are displayed separately. The 300
and 1000 update cohorts have different frame counts and are never pooled.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptok_hr.runtime import write_json
from experiments.adaptok_hr.fine_recovery.figure_qa import audit
from experiments.adaptok_hr.fine_recovery.figures import canvas

from .run import PROTOCOL, ROOT
from .study import read

COLORS = {"R00": "#55585C", "R10": "#007C83", "R01": "#9463A7", "R11": "#C36C2C",
          "Historical": "#243547", "C0": "#3364A0", "D0": "#55585C", "D1": "#007C83", "D2": "#9463A7", "D3": "#C36C2C"}
STYLES = {"R00": "--", "R10": "-", "R01": "-.", "R11": ":", "Historical": "-.",
          "C0": ":", "D0": "--", "D1": "-", "D2": "-.", "D3": ":"}
BUDGETS = ("256", "512", "1024", "2048")
FIGS = ROOT / "figures"


def native_plots():
    groups = {}
    for path in sorted(ROOT.glob("*/validation_*_*/summary.json")):
        run = path.parents[1].name
        seed, arm = run.split("_", 1)
        row = read(path)
        steps = row["steps"]
        groups.setdefault((seed, steps), {})[arm] = row
    return groups


def learning_curves(seed, stage, directories):
    training, validation, probes = {}, {}, {}
    for directory in directories:
        arm = directory.name.split("_", 1)[1]
        path = directory / "training.jsonl"
        if not path.is_file():
            continue
        start = 1000 if stage else 0
        with path.open() as file:
            mse = np.array([r["mse"] for line in file if (r := json.loads(line))["step"] > start])
        if len(mse) >= 25:
            pooled = np.convolve(mse, np.ones(25) / 25, "valid")
            training[arm] = (list(range(start + 25, start + len(mse) + 1)), (-10 * np.log10(pooled)).tolist())
        fast = sorted(directory.glob("validation_*.json"))
        if stage:
            parent = read(directory / "fork.json")["lineage"]["parent"]
            fast.insert(0, ROOT / parent / "validation_001000.json")
        validation[arm] = ([int(p.stem.split("_")[1]) for p in fast],
                           [read(p)["budgets"]["2048"]["psnr_global"] for p in fast])
        paths = sorted((directory / "probes").glob("*.json"))
        if paths:
            probes[arm] = ([int(p.stem) for p in paths],
                           [read(p)["levels"]["incremental_suffix"]["used_codes"] for p in paths])
    return {f"{seed}_{stage}{key}": ("Optimizer updates", label, lines) for key, label, lines in (
                ("training", "Training PSNR, 25-update mean MSE (dB)", training),
                ("validation", "K2048 short Validation global PSNR (dB)", validation),
                ("suffix_usage", "Used suffix IDs on the fixed Train I frame", probes)) if lines}


def series_data():
    data = {}
    for (seed, steps), results in native_plots().items():
        if steps >= 1000:
            results["Historical"] = read(PROTOCOL["historical_validation"])
        for key, label in (("psnr_global", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"),
                           ("dists", "DISTS")):
            lines = {}
            for arm, row in results.items():
                budgets = [row["closed_loop"][k] for k in BUDGETS]
                lines[arm] = ([s["actual_bpp"] for s in budgets], [s[key] for s in budgets])
            data[f"{seed}_{steps}_rd_{key}"] = ("Actual rate (bits / pixel)", label, lines)
        used = {arm: ([r["closed_loop"][k]["actual_bpp"] for k in BUDGETS],
                      [r["closed_loop"][k]["levels"]["Fine"]["used_codes"] for k in BUDGETS])
                for arm, r in results.items() if arm != "Historical"}
        data[f"{seed}_{steps}_sent_fine"] = ("Actual rate (bits / pixel)", "Distinct transmitted Fine IDs", used)
        for key, label in (("shuffle", "MSE increase after within-frame shuffle"),
                           ("cross", "MSE increase after same-position video swap")):
            lines = {arm: (list(range(3)), [r["information"]["sources"][s][key] for s in PROTOCOL["sources"]])
                     for arm, r in results.items() if arm != "Historical"}
            data[f"{seed}_{steps}_{key}"] = ("Validation source", label, lines)
    for seed in ("S1", "S2"):
        data.update(learning_curves(seed, "", sorted(ROOT.glob(seed + "_R*"))))
        data.update(learning_curves(seed, "decoder_", sorted(ROOT.glob(seed + "_[CD]*"))))
    return data


def draw(name, data, alignment):
    xlabel, ylabel, lines = data
    fig, ax = canvas(ylabel, xlabel)
    for label, (x, y) in lines.items():
        ax.plot(x, y, label=label, color=COLORS[label], linestyle=STYLES[label], linewidth=1.2,
                marker=None if name.endswith("_training") else "o", markersize=3)
    if xlabel == "Validation source":
        ax.set_xticks(range(3), ["Blackswan", "Car-roundabout", "Dog"])
        ax.axhline(0, linewidth=.5, color="#999999", zorder=0)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.21), ncol=min(6, len(lines)), frameon=False)
    fig.subplots_adjust(left=.15, right=.97, bottom=.19, top=.81)
    fig.canvas.draw()
    alignment(fig, json_out=str(FIGS / f"{name}.alignment.json"), strict=True)
    for ext in (".svg", ".pdf", ".png"):
        fig.savefig(FIGS / (name + ext), dpi=600)
    plt.close(fig)


def qa(scripts):
    source = audit([sys.executable, str(scripts / "validate_figure.py"), __file__, "--json"])
    checks = [source]
    for pdf in sorted(FIGS.glob("*.pdf")):
        if pdf.stem.endswith(".overlay"):
            continue
        for cmd in (["audit_pdf_text.py", str(pdf), "--min-pt", "5", "--json"],
                    ["audit_figure_collisions.py", str(pdf), "--json-out", str(pdf.with_suffix(".collision.json")),
                     "--overlay-pdf", str(pdf.with_suffix(".overlay.pdf"))]):
            checks.append(audit([sys.executable, str(scripts / cmd[0]), *cmd[1:]]))
    write_json(FIGS / "qa.json", checks)
    if any(r["returncode"] for r in checks):
        raise ValueError("figure audit requires attention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                               "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                               "pdf.fonttype": 42, "svg.fonttype": "none"})
    FIGS.mkdir(parents=True, exist_ok=True)
    source = series_data()
    write_json(FIGS / "plotted_values.json", source)
    for name, data in source.items():
        draw(name, data, require_matplotlib_panel_alignment)
    qa(args.qa_scripts)


if __name__ == "__main__":
    main()
