"""Scientific figures rendered directly from native-resolution C1 result rows."""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from aip_vtok.diagnostics.diag0.report import select

from .artifacts import OUTPUT
from .scan import BUDGETS

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "svg.hashsalt": "aip-vtok-c1", "pdf.fonttype": 42,
    "legend.frameon": False, "axes.linewidth": 0.75,
}
BLUE, GRAY, ORANGE = "#216589", "#6D7477", "#A7572C"


def axes():
    mpl.rcParams.update(STYLE)
    fig, axis = plt.subplots(figsize=(5.9, 3.45), layout="constrained")
    axis.grid(axis="y", linewidth=0.5, alpha=0.23)
    axis.set_axisbelow(True)
    return fig, axis


def export(fig, name):
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300)
    fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    svg = folder / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    plt.close(fig)


def rank_plot(summary):
    fig, axis = axes()
    for coordinate, label, color in (("aligned", "Aligned", BLUE), ("p_native", "Physical P", ORANGE)):
        for history, context, line in (("r0_aligned", "R0", "-"), ("fix_a_low_reference", "Fix-A", "--")):
            selected = select(summary, frame_type="P", history=history, coordinate=coordinate,
                              version="cont", fill="zero_fill")
            rows = sorted(selected, key=lambda r: r["rank"])
            ticks = list(range(len(rows)))
            axis.plot(ticks, [r["psnr_global_mse"] for r in rows], linestyle=line, color=color,
                      marker="o", markersize=4, label=f"{label} · {context}")
            axis.set_xticks(ticks, [str(r["rank"]) for r in rows])
    axis.set(xlabel="Retained channels (tested ranks)", ylabel="Global PSNR (dB)")
    axis.legend(ncols=2, loc="upper left")
    export(fig, "rank_p")


def budget_plot(summary, name):
    fig, axis = axes()
    styles = {
        ("B2", 0): (GRAY, "-", "o"), ("B2-ZeroFill", 0): (GRAY, "--", "s"),
        ("Fix-A", 0): (ORANGE, ":", "D"), ("B3", 1): (BLUE, "-", "o"), ("B3", 2): (BLUE, "--", "^"),
    }
    for (method, seed), (color, line, marker) in styles.items():
        rows = sorted((r for r in summary if r["method"] == method and r.get("seed", 0) == seed),
                      key=lambda r: r["budget"])
        if rows:
            label = f"{method}, seed {seed}" if seed else method
            axis.plot(range(len(rows)), [r["psnr_global_mse"] for r in rows], color=color,
                      linestyle=line, marker=marker, markersize=4, label=label)
    axis.set(xlabel="Displayed 6D slots", ylabel="Global PSNR (dB)",
             xticks=list(range(len(BUDGETS))), xticklabels=[str(k) for k in BUDGETS])
    axis.legend(ncols=2, loc="best")
    export(fig, name)


def dense_plot(gains, threshold):
    fig, axis = axes()
    for mode, label, color, marker in (("same_context_b0_fsq_history", "B0-FSQ history", BLUE, "o"),
                                       ("independent_closed", "Independent loops", ORANGE, "^")):
        rows = sorted((r for r in gains if r["history"] == mode), key=lambda r: r["seed"])
        offset = -0.08 if marker == "o" else 0.08
        axis.scatter([r["seed"] + offset for r in rows], [r["psnr_gain_db"] for r in rows],
                     color=color, marker=marker, s=32, label=label, zorder=4)
    axis.axhline(threshold, color=GRAY, linestyle="--", linewidth=1, label="PSNR gate")
    axis.set(xlabel="Training seed", ylabel="B1 − B0 global PSNR (dB)", xticks=(1, 2), xlim=(0.55, 2.45))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 0.88), ncols=3)
    export(fig, "dense_gain")
