"""Render the seven measured HLA-Tok-C figures and their alignment reports."""

import argparse
import os
import sys
from itertools import groupby
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.hla_tok.continuous.inputs import RESULTS
from scripts.hla_tok.v2.protocol import csv_at, json_at

METRIC_AXES = (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"))
ORDER_COLORS = {"O0": "#535e6c", "O1": "#a5631b", "O2": "#096b73"}
WIRE_MARKS = {"fp32": ("-", "^"), "fp16": ("--", "s"), "bf16": (":", "o")}
STAGE_COLORS = ("#316a8a", "#b4c1c9", "#db945d", "#765a8b", "#aa7898", "#2f8a7a")
WIDTH_IN = 183 / 25.4


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.65,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def grid(rows, columns, height, *, bottom=0.25, left=0.087, wspace=0.48, hspace=0.56):
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(WIDTH_IN, height))
    fig.subplots_adjust(left=left, right=0.978, bottom=bottom, top=0.89, wspace=wspace, hspace=hspace)
    for label, axis in zip(ascii_lowercase, axes.flat):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, size=8, weight="bold", va="bottom")
        axis.tick_params(length=2.5, pad=2)
    return fig, axes


def _rows(root: Path, *names: str) -> list[dict]:
    return [row for name in names for row in csv_at(root / name)]


def _group(rows, *keys):
    def key(row):
        return tuple(row[name] for name in keys)

    return [(labels, list(group)) for labels, group in groupby(sorted(rows, key=key), key)]


def _measure_series(axis, data, field, abscissa, **style_kwargs):
    rows = sorted(data, key=lambda row: float(row[abscissa]))
    xs = [float(row[abscissa]) for row in rows]
    if axis.get_xscale() == "log" and any(x <= 0 for x in xs):
        raise ValueError("logarithmic rate figures require positive measured rates")
    axis.plot(xs, [float(row[field]) for row in rows], **style_kwargs)


def _order_legend():
    return [Line2D([0], [0], color=color, linewidth=1.5, label=label) for label, color in ORDER_COLORS.items()]


def _wire_legend():
    return [Line2D([0], [0], color="#333333", linestyle=pattern, marker=marker, markersize=3.5,
                   markerfacecolor="none", label=dtype.upper()) for dtype, (pattern, marker) in WIRE_MARKS.items()]


def _legend(fig, handles, columns):
    fig.legend(handles=handles, ncol=columns, loc="lower center", bbox_to_anchor=(0.5, 0.008),
               columnspacing=1.5, handlelength=1.8)


def _metric_grid(x_label, *, legend_rows=1):
    height, bottom = (2.4, 0.34) if legend_rows == 2 else (2.02, 0.28)
    fig, axes = grid(1, 3, height, bottom=bottom)
    for axis, (_, text) in zip(axes.flat, METRIC_AXES, strict=True):
        axis.set_ylabel(text)
        axis.set_xlabel(x_label)
    return fig, axes.flat


def budget_distortion(root: Path):
    fixed = _rows(root, "fixed_budget.csv", "ordered_fixed_budget.csv")
    closed = _rows(root, "closed_budget.csv", "ordered_closed_budget.csv")
    fig, axes = _metric_grid("Continuous vectors (×1000)")
    for axis, (field, _) in zip(axes, METRIC_AXES, strict=True):
        for mode, data in (("FIXED", fixed), ("CLOSED", closed)):
            for (order,), rows in _group(data, "order"):
                normalized = [row | {"k": int(row["budget"]) / 1000} for row in rows]
                _measure_series(axis, normalized, field, "k", color=ORDER_COLORS[order],
                                linestyle="-" if mode == "FIXED" else "none", marker="o", markersize=3.7,
                                markerfacecolor=ORDER_COLORS[order] if mode == "FIXED" else "white",
                                markeredgewidth=0.8)
    _legend(fig, [*_order_legend(), Line2D([0], [0], color="#333333", marker="o", markersize=3.4,
                                          label="FIXED", linestyle="-"),
                  Line2D([0], [0], color="#333333", marker="o", markersize=3.4,
                         markerfacecolor="white", label="CLOSED", linestyle="none")], 5)
    return fig


def _format_series(axis, data, field, abscissa):
    for (order, dtype), rows in _group(data, "order", "wire_dtype"):
        linestyle, marker = WIRE_MARKS[dtype]
        _measure_series(axis, rows, field, abscissa, linewidth=1.05, color=ORDER_COLORS[order],
                        linestyle=linestyle, marker=marker, markersize=3.1, markerfacecolor="none")


def rate_distortion(root: Path):
    formal = _rows(root, "wire_rate_distortion.csv")
    equal = [row for row in _rows(root, "equal_rate_comparison.csv") if row["comparison"] == "equal_payload_cap"]
    measured = formal + [row for row in equal if row["method"] == "C0 continuous"]
    c1, = [row for row in equal if row["method"] == "C1 Fixed13"]
    original = _rows(root, "original_adaptok_context.csv")
    fig, axes = _metric_grid("Actual payload (bits/pixel)", legend_rows=2)
    for axis, (field, _) in zip(axes, METRIC_AXES, strict=True):
        axis.set_xscale("log")
        axis.set_xticks([0.02, 0.1, 0.5, 2, 8], ["0.02", "0.1", "0.5", "2", "8"])
        _format_series(axis, measured, field, "payload_bpp")
        axis.scatter(float(c1["payload_bpp"]), float(c1[field]), marker="D", s=22, color="#a03452", zorder=6)
        _measure_series(axis, original, field, "actual_bpp", marker="x", markersize=3.8, linewidth=0.65,
                        color="#8f979e", linestyle="--")
    external = [Line2D([0], [0], color="#a03452", marker="D", markersize=4.0, label="C1 Fixed13", linestyle="none"),
                Line2D([0], [0], color="#8f979e", marker="x", markersize=4.0, label="Original AdapTok · 16-frame",
                       linestyle="--")]
    _legend(fig, [*_order_legend(), *_wire_legend(), *external], 4)
    return fig


def rank_importance(root: Path):
    importance = _rows(root, "rank_importance_train.csv")
    gains = json_at(root / "ordering_decision.json")["signed_low_budget_gains"]
    fig, axes = grid(1, 2, 2.0, bottom=0.25, left=0.10, wspace=0.33)
    left, right = axes[0]
    left.plot([int(row["original_rank"]) for row in importance],
              [float(row["mean_delta_rgb_mse"]) for row in importance], color=ORDER_COLORS["O2"], linewidth=1)
    left.axhline(0, color="#aaaaaa", linewidth=0.5)
    left.set_xlabel("Original Rank")
    left.set_ylabel("Train RGB MSE increase")
    left.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for (name,), rows in _group(gains, "order"):
        series = [row | {"k": row["budget"] / 1000} for row in rows]
        _measure_series(right, series, "global_psnr", "k", color=ORDER_COLORS[name],
                        linewidth=1.3, marker="o", markersize=3, label=name)
    right.axhline(0, color="#aaaaaa", linewidth=0.5)
    right.set_xlabel("Continuous vectors (×1000)")
    right.set_ylabel("PSNR gain over O0 (dB)")
    right.legend(loc="upper left", ncol=2)
    return fig


def _budget_palette(budgets):
    return dict(zip(budgets, mpl.colormaps["viridis"](np.linspace(0.13, 0.87, len(budgets))), strict=True))


def temporal(root: Path):
    order = json_at(root / "ordering_decision.json")["primary_order"]
    fixed = [row for row in _rows(root, "fixed_budget_per_frame.csv", "ordered_fixed_budget_per_frame.csv")
             if row["order"] == order]
    closed = {(row["video"], row["budget"], row["frame"]): row for row in
              _rows(root, "closed_budget_per_frame.csv", "ordered_closed_budget_per_frame.csv")
              if row["order"] == order}
    for row in fixed:
        pair = closed[row["video"], row["budget"], row["frame"]]
        row["difference"] = float(pair["psnr"]) - float(row["psnr"])
    budgets = sorted({int(row["budget"]) for row in fixed})
    palette = _budget_palette(budgets)
    fig, axes = grid(3, 3, 4.65, bottom=0.16, hspace=0.54)
    metrics = (("psnr", "Frame PSNR (dB)"), ("lpips", "Frame LPIPS-Alex"), ("difference", "CLOSED − FIXED (dB)"))
    limits = {}
    for field, _ in metrics:
        values = [float(row[field]) for row in fixed]
        low, high = min(values), max(values)
        margin = max(0.03, 0.05 * (high - low))
        limits[field] = low - margin, high + margin
    for col, ((video,), frames) in enumerate(_group(fixed, "video")):
        axes[0, col].set_title(video.split(":")[-1], loc="center")
        for row_index, (field, ylabel) in enumerate(metrics):
            axis = axes[row_index, col]
            for (budget,), samples in _group(frames, "budget"):
                _measure_series(axis, samples, field, "frame", color=palette[int(budget)], linewidth=0.8)
            axis.set_ylim(*limits[field])
            if col == 0:
                axis.set_ylabel(ylabel)
            if row_index == 2:
                axis.set_xlabel("Frame index")
    handles = [Line2D([0], [0], color=color, linewidth=1, label=f"K{budget}") for budget, color in palette.items()]
    _legend(fig, handles, 8)
    return fig


def _separated_timelines(records, gap=4):
    offset = 0
    timelines = []
    for (_,), rows in _group(records, "video"):
        frames = [row | {"timeline": offset + int(row["frame"])} for row in rows]
        timelines.append(frames)
        offset += len(frames) + gap
    return timelines


def dynamic(root: Path):
    order = json_at(root / "ordering_decision.json")["primary_order"]
    records = [row for row in _rows(root, "dynamic_budget_per_frame.csv") if row["order"] == order]
    schedules = ("constant", "descending", "ascending", "alternate", "random", "recover")
    names = ("Constant", "Descending", "Ascending", "Alternating", "Seeded random", "Low then maximum")
    fig, axes = grid(6, 2, 7.0, bottom=0.082, left=0.135, wspace=0.33, hspace=0.59)
    psnr_values = [float(row["psnr"]) for row in records]
    low = np.floor(min(psnr_values) / 5) * 5
    high = np.ceil(max(psnr_values) / 5) * 5
    for i, (mode, title) in enumerate(zip(schedules, names, strict=True)):
        left, right = axes[i]
        left.set_title(title, loc="left", pad=3.0)
        for clip in _separated_timelines([row for row in records if row["schedule"] == mode]):
            _measure_series(left, [row | {"k": float(row["budget"]) / 1000} for row in clip],
                            "k", "timeline", color="#316a8a", drawstyle="steps-post", linewidth=0.9)
            _measure_series(right, clip, "psnr", "timeline", color="#176d66", linewidth=0.9)
        left.set(ylim=(0, 17), yticks=[0, 8, 16], ylabel="K (×1000)")
        right.set(ylim=(low, high), yticks=np.linspace(low, high, 3), ylabel="PSNR (dB)")
        if i == len(schedules) - 1:
            left.set_xlabel("Frames, with a gap between clips")
            right.set_xlabel("Frames, with a gap between clips")
        else:
            for axis in (left, right):
                axis.tick_params(labelbottom=False)
    return fig


def _latency_inputs(rows):
    stages = {
        "Encoder": ("encoder",),
        "Order and casts": ("ordering_and_truncation", "wire_dtype_conversion", "decoder_dtype_conversion"),
        "Transfers": ("gpu_to_cpu", "cpu_to_gpu"),
        "Serialize": ("serialization",),
        "Deserialize": ("deserialization",),
        "Decode": ("recovery_network", "rgb_decoder"),
    }
    keys = sorted({int(row["budget"]) for row in rows})
    measured = {(int(row["budget"]), row["stage"]): float(row["mean_ms"]) for row in rows}
    stacks = [(label, np.array([sum(measured[key, stage] for stage in wanted) for key in keys]))
              for label, wanted in stages.items()]
    return keys, stacks, [measured[key, "total_serial"] for key in keys]


def latency(root: Path):
    rows = [row for row in _rows(root, "latency_breakdown.csv") if row["path"] == "float_wire_serial"]
    fig, axes = grid(2, 3, 3.7, bottom=0.18, hspace=0.59)
    covers: list = []
    total_max = max(float(row["mean_ms"]) for row in rows if row["stage"] == "total_serial")
    for i, precision in enumerate(("fp32", "bf16")):
        for j, dtype in enumerate(WIRE_MARKS):
            axis = axes[i, j]
            keys, stacks, total = _latency_inputs([
                row for row in rows if row["compute_dtype"] == precision and row["wire_dtype"] == dtype
            ])
            x = np.arange(len(keys))
            bottom = np.zeros(len(keys))
            covers = []
            for (label, values), color in zip(stacks, STAGE_COLORS, strict=True):
                bars = axis.bar(x, values, bottom=bottom, color=color, width=0.76, label=label)
                bottom += values
                covers.append(bars)
            point, = axis.plot(x, total, marker="_", linestyle="none", color="#1b1b1b", markersize=8,
                               label="Serial wall time")
            axis.set(title=f"{precision.upper()} compute · {dtype.upper()} wire", ylim=(0, 1.16 * total_max))
            axis.set_xticks([0, 2, 4, 7], [f"{keys[k] / 1000:g}" for k in (0, 2, 4, 7)])
            if j == 0:
                axis.set_ylabel("Mean latency (ms)")
            if i == 1:
                axis.set_xlabel("Continuous vectors (×1000)")
    _legend(fig, [*covers, point], 7)
    return fig


def bandwidth(root: Path):
    rows = [row for row in _rows(root, "bandwidth_constrained_quality.csv") if row["rate_feasibility"] == "FEASIBLE"]
    fig, axes = _metric_grid("Available bandwidth at 30 fps (Mbps)")
    for axis, (field, _) in zip(axes, METRIC_AXES, strict=True):
        _format_series(axis, rows, field, "target_mbps")
        axis.set_xticks([5, 10, 20, 40])
    names = {row["order"] for row in rows}
    _legend(fig, [*(item for item in _order_legend() if item.get_label() in names), *_wire_legend()], 5)
    return fig


FIGURES = {
    "01_budget_distortion": budget_distortion,
    "02_rate_distortion": rate_distortion,
    "03_rank_importance": rank_importance,
    "04_temporal_stability": temporal,
    "05_dynamic_budget": dynamic,
    "06_latency_breakdown": latency,
    "07_bandwidth_quality": bandwidth,
}


def render(output: Path, qa: Path, names: list[str]) -> None:
    sys.path.insert(0, str(qa))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    style()
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = FIGURES[name](output)
        basename = destination / name
        require_matplotlib_panel_alignment(fig, json_out=basename.with_suffix(".alignment.json"),
                                           require_panel_labels=True, strict=True)
        for suffix in (".pdf", ".svg", ".png"):
            fig.savefig(basename.with_suffix(suffix), dpi=300)
        plt.close(fig)
        print(f"Rendered {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--qa", type=Path, default=Path(os.environ.get(
        "NATURE_FIGURE_SCRIPTS", Path.home() / ".codex/skills/nature-figure/scripts")))
    parser.add_argument("--figure", action="append", choices=FIGURES)
    args = parser.parse_args()
    render(args.output, args.qa, args.figure or list(FIGURES))


if __name__ == "__main__":
    main()
