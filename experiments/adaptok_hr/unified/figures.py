"""Actual packet rate versus native quality; seeds and milestones remain separate.

Contract: quantify whether objective or architecture changes improve all four
transmitted budgets. Each seed is one trained replicate; correlated video frames
are not independent replicates, so no invented error bars are drawn. All points
and source hashes are exported. Training curves use a trailing 25-update mean
of main-budget MSE, then convert to PSNR; they are not native Validation curves.
Single quantitative panels reuse only the
repository's Python typography/canvas/export conventions, not scientific data.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from adaptok_hr.runtime import file_hash, write_json
from experiments.adaptok_hr.fine_recovery.figure_qa import audit
from experiments.adaptok_hr.fine_recovery.figures import canvas, export

from .evaluate import read_rows
from .run import ROOT

COLORS = {"U0": "#55585C", "U1": "#007C83", "U2": "#9463A7", "U3": "#C36C2C",
          "N0": "#55585C", "N1": "#007C83", "N2": "#C36C2C"}
STYLES = {"U0": "--", "U1": "-", "U2": "-.", "U3": ":", "N0": "--", "N1": "-", "N2": "-."}
LABELS = {"psnr_global": "Global PSNR (dB)", "lpips": "Native LPIPS-Alex (lower is better)",
          "dists": "Native DISTS (lower is better)", "cb_mae": "Cb MAE (BT.709 full range)",
          "cr_mae": "Cr MAE (BT.709 full range)",
          "color_difference_variance_retention": "Chroma spatial variance retained"}


def measured_groups():
    groups = {}
    for path in sorted(ROOT.glob("S*/validation_*_64/summary.json")):
        name = path.parents[1].name
        seed, arm = name.split("_", 1)
        row = json.loads(path.read_text())
        groups.setdefault((seed, arm[0], row["completed_steps"]), {})[arm] = (path, row)
    return groups


def rate_distortion(output, alignment):
    source = {}
    for (seed, stage, step), arms in measured_groups().items():
        for metric, ylabel in LABELS.items():
            figure, axis = canvas(ylabel, "Actual bitstream rate (bits / pixel)")
            for arm, (path, row) in arms.items():
                values = [row["closed_loop"][str(k)] for k in (256, 512, 1024, 2048)]
                xx, yy = [v["actual_bpp"] for v in values], [v[metric] for v in values]
                axis.plot(xx, yy, label=arm, color=COLORS[arm], linestyle=STYLES[arm],
                          linewidth=1.3, marker="o", markersize=3)
                source[f"{seed}_{stage}_{step}_{metric}/{arm}"] = {"x": xx, "y": yy,
                                            "source": str(path), "source_sha256": file_hash(path)}
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            export(figure, axis, f"{seed}_{stage}_{step}_rd_{metric}", output, alignment)
    return source



def training_curve(path, stage):
    rows = [r for r in read_rows(path) if r["step"] > (1300 if stage == "U" else 0)]
    if len(rows) < 25:
        return [], []
    x = [r["step"] for r in rows][24:]
    mean = np.convolve([r["mse"] for r in rows], np.ones(25)/25, "valid")
    return x, (-10*np.log10(mean)).tolist()


def learning_curves(output, alignment):
    source = {}
    for seed in ("S1", "S2"):
        for stage in ("U", "N"):
            paths = sorted(ROOT.glob(f"{seed}_{stage}*/training.jsonl"))
            if not paths:
                continue
            figure, axis = canvas("Training global PSNR (dB)")
            for path in paths:
                arm = path.parent.name.split("_", 1)[1]
                x, yy = training_curve(path, stage)
                if not x:
                    continue
                axis.plot(x, yy, label=arm, color=COLORS[arm], linestyle=STYLES[arm], linewidth=1.1)
                source[f"{seed}_{stage}_learning/{arm}"] = {"x": x, "y": yy, "source": str(path),
                                                                    "source_sha256": file_hash(path)}
            export(figure, axis, f"{seed}_{stage}_training", output, alignment)
    return source


def qa(output, scripts):
    resolved = output / "plotting_source_for_audit.py"
    resolved.write_text(Path(__file__).read_text() + "\n" + inspect.getsource(canvas) + "\n" + inspect.getsource(export))
    records = [audit([sys.executable, str(scripts / "validate_figure.py"), str(resolved), "--json"])]
    for pdf in sorted(output.glob("*.pdf")):
        if "overlay" in pdf.name:
            continue
        for command in (["audit_pdf_text.py", str(pdf), "--min-pt", "5", "--json"],
                        ["audit_figure_collisions.py", str(pdf), "--json-out", str(pdf.with_suffix(".collision.json"))]):
            records.append({"file": str(pdf), **audit([sys.executable, str(scripts / command[0]), *command[1:]])})
    write_json(output / "qa.json", records)
    if any(r["returncode"] for r in records):
        raise ValueError("figure source/text/collision QA did not pass")


def configure_style():
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                               "font.size": 8, "axes.labelsize": 8,
                               "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
                               "svg.fonttype": "none", "pdf.fonttype": 42})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    configure_style()
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source = {**rate_distortion(output, require_matplotlib_panel_alignment),
              **learning_curves(output, require_matplotlib_panel_alignment)}
    write_json(output / "source_data.json", source)
    (output / "figure_contract.md").write_text(__doc__ + "\nNo exclusions. All four budgets and both seeds are shown separately.\n")
    qa(output, args.qa_scripts)
