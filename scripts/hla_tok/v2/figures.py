"""Reproducible scientific figures from the measured V2 CSV tables."""

# ruff: noqa: E402
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.hla_tok.v2.protocol import OUTPUT, ROOT, csv_at

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 180, "savefig.dpi": 300,
})

COLOR = dict(R0="#647080", R1="#7866A5", R2="#087A72", R3="#BC6231")
ARM = ("R0", "R1", "R2", "R3")
DIAG = OUTPUT / "diag0"
FIGS = OUTPUT / "figures"


def emit(fig, path, labels=True, exclude_axes=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(fig.axes) - len(exclude_axes) > 1:
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig, json_out=str(path.with_suffix(".alignment.json")), exclude_axes=list(exclude_axes),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=labels, strict=True,
        )
    for extension in (".pdf", ".svg", ".png"):
        fig.savefig(path.with_suffix(extension))
    plt.close(fig)


def panel_label(axis, index):
    axis.annotate(chr(97 + index), (0, 1), xycoords="axes fraction", xytext=(-9, 6),
                  textcoords="offset points", fontweight="bold", va="bottom")


def attention():
    rows = [r for r in csv_at(DIAG / "attention_metrics.csv") if r["stage"] == "fixed"]
    fig, axes = plt.subplots(2, 3, figsize=(6.1, 3.55), layout="constrained")
    matrices = {}
    for layer in (1, 2):
        for start in (0, 16, 64):
            points = [r for r in rows if int(r["start_rank"]) == start and int(r["layer"]) == layer]
            matrices[layer, start] = np.array([mean(float(r[f"p{i}"]) for r in points)
                                              for i in range(64)]).reshape(8, 8)
    bound = max(data.max() for data in matrices.values())
    for index, (axis, ((layer, start), data)) in enumerate(zip(axes.flat, matrices.items())):
        image = axis.imshow(data, cmap="viridis", vmin=0, vmax=bound, interpolation="nearest")
        axis.set(xticks=[0, 3, 7], yticks=[0, 3, 7])
        axis.set_title(f"Rank {start}–{(15, 63, 119)[index % 3]}")
        if index % 3 == 0:
            axis.set_ylabel(f"Layer {layer}\nFeature grid y")
        if index >= 3:
            axis.set_xlabel("Feature grid x")
        panel_label(axis, index)
    colorbar = fig.colorbar(image, ax=axes, label="Mean attention mass", shrink=.83, pad=.02)
    emit(fig, DIAG / "figures/attention_coverage", exclude_axes=(colorbar.ax,))


def rank_spectrum():
    data = csv_at(DIAG / "rank_spectrum.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.0), layout="constrained")
    for stage, label, color in (("continuous", "R0 continuous", "#778398"), ("fixed", "R0 fixed VQ", "#087A72")):
        spectra = [r for r in data if r["stage"] == stage]
        y = [mean(float(r[f"s{k}"]) for r in spectra) for k in range(1, 33)]
        ax.plot(range(1, 33), y, label=label, color=color)
    ax.set(xlabel="Singular-value index", ylabel="Mean centered singular value", yscale="log")
    ax.legend(frameon=False)
    emit(fig, DIAG / "figures/centered_spectrum")


def intervention_change():
    rows = [r for r in csv_at(DIAG / "intervention_summary.csv") if r["stage"] == "fixed" and r["intervention"] != "original"]
    order = [r["intervention"] for r in rows if r["mode"] == "continuous"]
    labels = ["Mean", "Repeat rank 0", "Replace late", "Shuffle values", "Permute rank",
              "Delete first 16", "Delete mid 16", "Delete last 16"]
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 3.45), layout="constrained")
    for i, (ax, (key, title)) in enumerate(zip(axes, (
            ("global_psnr_delta", "Δ global PSNR (dB)"), ("delta_lpips", "Δ LPIPS-Alex"),
            ("delta_dists", "Δ DISTS")))):
        for mode, offset, color in (("continuous", -.15, "#778398"), ("quantized", .15, "#087A72")):
            vals = {row["intervention"]: float(row[key]) for row in rows if row["mode"] == mode}
            ax.plot([vals[x] for x in order], np.arange(len(order)) + offset, linestyle="none",
                    marker="o" if mode == "continuous" else "s", ms=3.4, color=color, label=mode.capitalize())
        ax.axvline(0, color="#C3C6CC", lw=.75, zorder=0)
        ax.set(xlabel=title, yticks=range(len(order)), yticklabels=labels if i == 0 else [])
        ax.set_ylim(len(order)-.45, -.55)
        panel_label(ax, i)
    axes[-1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.02), ncol=1)
    emit(fig, DIAG / "figures/token_interventions")


def draw_curves(by_arm, output):
    fields = (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"),
              ("effective_rank", "Effective rank"), ("centered_variance", "Centered variance"),
              ("raw_adjacent_cosine", "Adjacent cosine"))
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.1))
    fig.subplots_adjust(left=.102, right=.986, top=.855, bottom=.13, wspace=.43, hspace=.52)
    for i, (ax, (key, ylabel)) in enumerate(zip(axes.flat, fields)):
        for arm, data in by_arm.items():
            ax.plot([int(row["step"]) for row in data], [float(row[key]) for row in data], label=arm,
                    marker="o", markersize=2.5, linewidth=1.25, color=COLOR[arm])
        ax.set_ylabel(ylabel)
        if i >= 3:
            ax.set_xlabel("Optimizer step")
        panel_label(ax, i)
    axes[0, 2].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.04))
    emit(fig, output)

def learning_curves():
    seed = OUTPUT / "continuous_pilot/seed1"
    if not (seed / "metrics.csv").exists():
        return
    observed = {arm: csv_at(seed / arm / "pilot/training_curve.csv") for arm in ARM}
    draw_curves({arm: [r for r in rows if int(r["step"]) <= 1000] for arm, rows in observed.items()},
                FIGS / "matched_training_curves")
    for folder in sorted(seed.glob("extension_*")):
        if (folder / "metrics.csv").exists():
            arm = folder.name.rsplit("_", 1)[-1]
            continued = {key: [r for r in observed[key] if int(r["step"]) >= 1000] for key in ("R0", arm)}
            draw_curves(continued, FIGS / "matched_extension_curves")


def actual_rd():
    old = csv_at(ROOT / "results/hla_tok/baselines/metrics.csv")
    grouped = defaultdict(list)
    for row in old:
        if row["method"] == "Original AdapTok":
            name = f"AdapTok {row['stride']} / {row['precision'].upper()}"
        else:
            name = "Joint V1"
        grouped[name].append(row)
    row, = csv_at(ROOT / "results/hla_tok/phase1/fixed/validation_metrics.csv")
    grouped["R0 fixed"] = [row]
    for path in sorted((OUTPUT / "native").glob("seed1/*/native/validation/validation_metrics.csv")):
        chosen, = csv_at(path)
        grouped[f"{path.parents[2].name} fixed"] = [chosen]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.35), layout="constrained")
    sequence = ("#556579", "#A0A7B2", "#8875A1", "#B1A1BE", "#AF663D", "#202124", "#087A72")
    for i, (axis, (metric, ylabel)) in enumerate(zip(axes, (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex")))):
        for (name, rows), color in zip(grouped.items(), sequence):
            rows.sort(key=lambda r: float(r["actual_bpp"]))
            axis.plot([float(r["actual_bpp"]) for r in rows], [float(r[metric]) for r in rows],
                      label=name, marker="o", markersize=2.3, color=color, linewidth=1.0)
        axis.set(xlabel="Actual bits per source pixel", ylabel=ylabel)
        panel_label(axis, i)
    axes[-1].legend(frameon=False, ncol=2, bbox_to_anchor=(.5, 1.04), loc="lower center")
    emit(fig, FIGS / "actual_rate_distortion")


def latency():
    files = sorted((OUTPUT / "benchmark/seed1").glob("*/components.csv"))
    if not files:
        return
    selected = [row for path in files for row in csv_at(path)
                if row["precision"] == "bf16" and int(row["budget"]) == 16320]
    stages = ("spatial_encoder", "query_prepare", "local_query", "vq_search", "local_decoder",
              "cross_region_fusion", "rgb_decoder", "fixed13_and_id_transfer")
    short = ("Spatial encoder", "Query prepare", "Local query", "VQ search", "Local decoder",
             "Fusion", "RGB decoder", "Fixed13 and IDs")
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    for idx, arm in enumerate(ARM):
        costs = {row["stage"]: float(row["cuda_span_ms"] or 0) for row in selected if row["arm"] == arm}
        if costs:
            ax.plot([costs.get(stage, 0) for stage in stages], np.arange(len(stages)) + (idx - 1.5) * .15,
                    linestyle="none", marker="o", markersize=3, label=arm, color=COLOR[arm])
    ax.set(xlabel="Profiled CUDA scope (ms)", yticks=range(len(stages)), yticklabels=short)
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1))
    emit(fig, FIGS / "latency_breakdown")


def repeated_codes():
    import json

    prior = ROOT / "results/hla_tok/phase1/fixed/codebook.json"
    paths = [("R0 fixed", prior)]
    paths += [(f"{p.parents[2].name} fixed", p) for p in sorted(
        (OUTPUT / "native").glob("seed1/*/native/validation/codebook.json"))]
    values = [100 * json.loads(path.read_bytes())["16320"]["same_code_fraction"] for _, path in paths]
    fig, axis = plt.subplots(figsize=(4.1, 2.7), layout="constrained")
    axis.bar([name for name, _ in paths], values, color=["#647080", "#087A72"][:len(paths)], width=.48)
    axis.set(ylabel="Adjacent same-code fraction (%)", ylim=(0, 100))
    emit(fig, FIGS / "adjacent_same_code")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-tools", required=True, type=Path)
    parser.add_argument("--d0-only", action="store_true")
    args = parser.parse_args()
    sys.path.append(str(args.alignment_tools))
    attention()
    rank_spectrum()
    intervention_change()
    if not args.d0_only:
        learning_curves()
        actual_rd()
        repeated_codes()
        latency()


if __name__ == "__main__":
    main()
