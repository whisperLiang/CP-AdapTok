"""Plot measured PHT rate, quality, throughput and deadline outcomes."""

import csv
import json
from statistics import mean

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from experiments.pht_tok.evidence import OUT

PLOT = OUT / "figures"
INK = "#252a32"
BLUE = "#1856a3"
PALE = "#6b95c7"
GREEN = "#267866"
GREY = "#84868b"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.65,
    "legend.frameon": False, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
})


def save(fig, name):
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(PLOT / f"{name}.alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(PLOT / f"{name}.pdf")
    svg = PLOT / f"{name}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(PLOT / f"{name}.png", dpi=600)
    plt.close(fig)


def subset(rows, method, stage):
    return sorted((r for r in rows if r["method"] == method and int(r["stage"]) == stage),
                  key=lambda row: int(row["qp"]))


def rd(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.87, wspace=0.36, hspace=0.42)
    designs = [
        ("bpp", "global_psnr", "Physical rate (bits/pixel)", "Pooled PSNR (dB)"),
        ("mbps", "global_psnr", "Physical rate (Mbit/s)", "Pooled PSNR (dB)"),
        ("bpp", "lpips", "Physical rate (bits/pixel)", "LPIPS-Alex"),
        ("bpp", "dists", "Physical rate (bits/pixel)", "DISTS"),
    ]
    official = [r for r in subset(rows, "official", 4) if r["origin"] == "archived official diagonal QP"]
    choices = [("HT-S", official, INK, "-", ""),
               ("P1 prefix 1", subset(rows, "MASK", 1), PALE, "--", "o"),
               ("P1 prefix 2", subset(rows, "MASK", 2), BLUE, ":", "s"),
               ("P1 prefix 3", subset(rows, "MASK", 3), BLUE, "--", "^"),
               ("P1 Full", subset(rows, "MASK", 4), GREEN, "-", "o")]
    for panel, (ax, (x, y, xlabel, ylabel)) in enumerate(zip(axes.ravel(), designs, strict=True)):
        for label, series, color, style, mark in choices:
            rates = [float(r[x]) for r in series]
            if min(rates) <= 0:
                raise ValueError("logarithmic rate axes require positive physical bytes")
            ax.plot(rates, [float(r[y]) for r in series],
                    label=label, color=color, linestyle=style, linewidth=1, marker=mark, markersize=3,
                    markerfacecolor="white")
        ax.set_xscale("log")
        ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10) if x == "mbps" else (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
        ax.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.text(-0.16, 1.03, "abcd"[panel], transform=ax.transAxes, weight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.52, 0.98))
    save(fig, "rate_distortion")


def progressive(diagnosis):
    curves = diagnosis["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.19, top=0.81, wspace=0.37)
    for label, method, color, marker in (("P1 spatial groups", "MASK", BLUE, "o"),
                                         ("P2 precision refinement", "PRECISION", GREEN, "s")):
        for qp, fmt in ((16, ":"), (32, "-"), (48, "--")):
            series = sorted((row for row in curves if row["method"] == method and row["qp"] == qp),
                            key=lambda row: row["stage"])
            axes[0].plot([r["stage"] for r in series], [r["global_psnr"] for r in series],
                         marker=marker, color=color, markersize=3, linestyle=fmt, linewidth=1,
                         label=f"{label}, QP {qp}")
        overhead = sorted((r for r in diagnosis["full_overhead"] if r["method"] == method), key=lambda row: row["qp"])
        axes[1].plot([r["qp"] for r in overhead], [100 * r["extra_ratio"] for r in overhead],
                     marker=marker, color=color, markersize=3, linewidth=1)
    axes[0].set(xlabel="Completed quality prefixes", ylabel="Pooled PSNR (dB)", xticks=[1, 2, 3, 4])
    axes[1].set(xlabel="QP", ylabel="Full byte overhead (%)", xticks=[0, 16, 32, 48, 63])
    axes[1].axhline(5, linestyle=":", color=GREY, linewidth=0.7)
    axes[1].text(2, 5.22, "5% research target", color=INK, fontsize=6)
    for ax, letter in zip(axes, "ab", strict=True):
        ax.text(-0.17, 1.03, letter, transform=ax.transAxes, weight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.995))
    save(fig, "progression_and_overhead")


def performance(profiles):
    labels = ("Native HT-S", "PHT Base", "PHT Full", "PHT incremental")
    sample = [profiles["official_repetitions"]] + [profiles["mask_kref2"][mode]
                                                   for mode in ("base", "full", "incremental")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.88, wspace=0.34)
    colors = [GREY, PALE, BLUE, GREEN]
    for ax, metric, label, letter in zip(axes, ("encode_fps", "decode_fps"),
                                        ("Encode throughput (frames/s)", "Decode throughput (frames/s)"),
                                        "ab", strict=True):
        values = [mean(row[metric] for row in group) for group in sample]
        ax.bar(range(len(values)), values, color=colors, width=0.61)
        for i, scores in enumerate(sample):
            ax.scatter([i] * len(scores), [row[metric] for row in scores], s=7, linewidth=0, color=INK, zorder=3)
        ax.axhline(values[0] * 0.7, color=INK, linestyle=":", linewidth=0.8)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel=label, ylim=(0, max(values) * 1.15))
        ax.tick_params(axis="x", labelrotation=27)
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes, weight="bold")
    save(fig, "throughput")


def fixed_closed(open_loop, closed):
    curves = open_loop["curves"]
    fixed = sorted((row for row in curves if row["method"] == "MASK" and row["stage"] >= 2),
                   key=lambda row: row["stage"])
    r1 = []
    for pattern in ("base", "mid", "full"):
        r1.append(next(row for row in closed["runs"] if (row["mode"], row["k_ref"], row["pattern"]) == (
            "MASK", 2, pattern)))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.19, right=0.95, bottom=0.2, top=0.88)
    ax.plot([r["mbps"] for r in fixed], [r["global_psnr"] for r in fixed],
            color=GREY, marker="s", label="FIXED official reference")
    ax.plot([r["physical_bytes"] * 8 * r["source_fps"] / r["source_frames"] / 1e6 for r in r1],
            [r["quality"]["global_psnr"] for r in r1], color=BLUE, marker="o", label="CLOSED R1 prefix 2")
    ax.set(xlabel="Physical rate (Mbit/s)", ylabel="Pooled RGB PSNR (dB)")
    ax.legend(loc="best")
    save(fig, "fixed_vs_closed")


def network(report):
    cases = list(dict.fromkeys(row["trace"] for row in report["runs"] if not row["trace"].startswith("static_")))
    labels = ["High/low", "Burst", "Sawtooth", "Random + loss", "Verizon LTE"][:len(cases)]
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.15))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.87, hspace=0.39)
    for policy, legend, color, marker in (("S0_fixed_qp", "S0 fixed QP", GREY, "o"),
                                           ("S1_adaptive_qp", "S1 adaptive QP", INK, "s"),
                                           ("S4_adaptive_prefix", "S4 adaptive PHT prefix", BLUE, "^")):
        rows = [next(r for r in report["runs"] if r["policy"] == policy and r["trace"] == case) for case in cases]
        for ax, field, factor in zip(axes, ("deadline_hit_rate", "qoe_drop_penalized_psnr"), (100, 1), strict=True):
            ax.plot(range(len(cases)), [r[field] * factor for r in rows], lw=1, color=color, marker=marker,
                    markersize=4, label=legend)
    axes[0].set(ylabel="On-time frames (%)", ylim=(-5, 105), xticks=range(len(cases)), xticklabels=[])
    axes[1].set(ylabel="Drop-penalized PSNR score", xticks=range(len(cases)), xticklabels=labels)
    for ax, letter in zip(axes, "ab", strict=True):
        ax.text(-0.13, 1.03, letter, transform=ax.transAxes, weight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.51, 0.98))
    save(fig, "dynamic_network")


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    PLOT.mkdir(exist_ok=True)
    with (OUT / "rate_distortion.csv").open() as h:
        rd(list(csv.DictReader(h)))
    progressive(read("progressive_diagnosis.json"))
    performance(read("profiling.json"))
    if (OUT / "fixed_uvg.json").is_file():
        fixed_closed(read("fixed_uvg.json"), read("reference_results.json"))
    if (OUT / "network_results.json").is_file():
        network(read("network_results.json"))


if __name__ == "__main__":
    main()
