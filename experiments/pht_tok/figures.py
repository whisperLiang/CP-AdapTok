"""Render audited figures from the unified PHT-Tok experiment ledgers."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "pht-tok",
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "legend.frameon": False,
})

COLORS = {"ld": "#126782", "hts": "#b06731"}
MARKERS = {"ld": "o", "hts": "s"}
METHODS = {
    "native_ld": "Native LD", "pht2_ld_r1": "PHT2 LD", "native_hts": "Native HT-S",
    "original_pht1_hts_at_54674cc": "Original PHT1", "pht1_hts": "Unified PHT1", "pht2_hts_r1": "PHT2 HT-S",
}


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="semibold", pad=7)


def save(fig, folder, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(folder / f"{name}.alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(folder / f"{name}.svg", metadata={"Date": None})
    fig.savefig(folder / f"{name}.png", dpi=600)
    plt.close(fig)


def points(rows, method):
    return [row for row in rows if row["method"] == method]


def variables(rows, x, y):
    return [float(row[x]) for row in rows], [float(row[y]) for row in rows]


def plot_rd(folder, rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.84, wspace=.40, hspace=.54)
    measures = (
        ("a", "Distortion", "global_psnr", "Pooled PSNR (dB)"),
        ("b", "Perceptual distance", "lpips", "LPIPS-Alex"),
        ("c", "Structural and texture distance", "dists", "DISTS"),
    )
    for ax, (letter, title, field, label) in zip(axes.flat, measures, strict=False):
        panel(ax, letter, title)
        for backend, color in COLORS.items():
            native = sorted(points(rows, f"native_{backend}"), key=lambda r: float(r["mbps"]))
            closed = points(rows, f"pht2_{backend}_r1")
            rates, values = variables(native, "mbps", field)
            if min(rates) <= 0 or any(float(r["mbps"]) <= 0 for r in closed):
                raise ValueError("log axes require positive measured physical rates")
            ax.plot(rates, values, color=color, marker=MARKERS[backend], ms=3, lw=1.3,
                    label=f"Native {'LD' if backend == 'ld' else 'HT-S'}")
            ax.scatter(*variables(closed, "mbps", field), c=color, alpha=.42, s=13,
                       marker=MARKERS[backend], label=f"PHT2 {'LD' if backend == 'ld' else 'HT-S'}")
        ax.set(xscale="log", xlabel="Physical rate (Mbps)", ylabel=label)
    ax = axes[1, 1]
    panel(ax, "d", "All eight prefixes at QP 32")
    for backend, color in COLORS.items():
        closed = sorted((r for r in points(rows, f"pht2_{backend}_r1") if r["qp"] == "32"),
                        key=lambda r: int(r["stage"]))
        ax.plot(*variables(closed, "bytes_per_frame", "global_psnr"), lw=1.4,
                ms=3.2, marker=MARKERS[backend], c=color)
    ax.set(xlabel="Physical prefix bytes per frame", ylabel="Pooled PSNR (dB)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.54, .97))
    save(fig, folder, "rate_distortion")


def plot_runtime(folder, throughput, overhead):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
    fig.subplots_adjust(left=.13, bottom=.25, right=.985, top=.84, wspace=.53)
    chosen = {row["method"]: row for row in throughput}
    selected = [chosen[method] for method in METHODS]
    palette = [COLORS["ld"], COLORS["ld"], COLORS["hts"], "#7c818a", "#7c818a", COLORS["hts"]]
    names = list(METHODS.values())
    y = np.arange(len(names))
    speed = [float(row["serial_fps"]) for row in selected]
    axes[0].barh(y, speed, color=palette)
    axes[0].axvline(30, color="#3c4148", linewidth=.9, linestyle=":")
    axes[0].set(yticks=y, yticklabels=names, xlabel="Serial throughput (FPS)")
    axes[0].invert_yaxis()
    panel(axes[0], "a", "Real codec throughput")

    timing_rows = [row for row in selected if row.get("first_display_ms_p50")]
    labels = [METHODS[row["method"]] for row in timing_rows]
    pos = np.arange(len(labels))
    wait = [float(r["capture_wait_ms_p50"]) for r in timing_rows]
    elapsed = [float(r["first_display_ms_p50"]) for r in timing_rows]
    axes[1].barh(pos, wait, color="#bfc9d1", height=.64, label="Capture")
    axes[1].barh(pos, np.asarray(elapsed) - wait, left=wait, color="#435667", height=.64,
                 label="Codec")
    axes[1].set(yticks=pos, yticklabels=labels, xlabel="Median time (ms)")
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2)
    panel(axes[1], "b", "First displayed picture")

    for backend, color in COLORS.items():
        series = [row for row in overhead if row["backend"] == backend]
        x, values = variables(series, "qp", "full_extra_percent")
        axes[2].plot(x, values, marker=MARKERS[backend], color=color, ms=3.5, lw=1.3)
    axes[2].axhline(2, color="#3c4148", linewidth=.9, linestyle=":")
    axes[2].set(xlabel="QP", ylabel="Added wire bytes (%)", xticks=[16, 32, 48, 63])
    panel(axes[2], "c", "Same-integer overhead")
    save(fig, folder, "runtime")


def plot_network(folder, table, ledger):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    fig.subplots_adjust(left=.16, bottom=.25, right=.97, top=.83, wspace=.48)
    traces = ("stable", "pre_encode_change", "post_encode_drop", "recovery", "random")
    labels = ("Stable", "Before encode", "After encode", "Recovery", "Random")
    index = {(row["trace"], row["method"]): float(row["drop_penalized_psnr_db"]) for row in table}
    for ax, (backend, color), letter in zip(axes, COLORS.items(), "ab", strict=True):
        panel(ax, letter, "LD frames" if backend == "ld" else "HT-S chunks")
        method = f"pht2_{backend}_r1"
        native = f"native_{backend}"
        ax.axvline(0, color="#535963", linewidth=.9)
        for i, trace in enumerate(traces):
            stats = ledger["paired_video_intervals"][trace][backend]
            low, high = stats["bootstrap_95_db"]
            ax.hlines(i, low, high, linewidth=1.05, color="#9199a2")
            effect = index[trace, method] - index[trace, native]
            oracle = index[trace, f"{method}_oracle"] - index[trace, native]
            ax.plot(effect, i, marker="o", color=color, ms=3.5, label="Online" if i == 0 else "")
            ax.plot(oracle, i, marker="D", markerfacecolor="white", color=color, ms=3.8,
                    label="Clairvoyant comparison" if i == 0 else "")
        ax.set(yticks=np.arange(5), yticklabels=labels, xlabel="Delivery score change (dB)")
        ax.invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(.62, -.31), ncol=2)
    save(fig, folder, "dynamic_network")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    source = args.results
    folder = source / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plot_rd(folder, csv_rows(source / "rate_distortion.csv"))
    plot_runtime(folder, csv_rows(source / "throughput.csv"), csv_rows(source / "ablation.csv"))
    plot_network(folder, csv_rows(source / "dynamic_network.csv"),
                 json.loads((source / "dynamic_evidence.json").read_text()))


if __name__ == "__main__":
    main()
