"""Single-axis scientific plots from the same evidence used in the report."""

import matplotlib as mpl
import matplotlib.pyplot as plt

from aip_vtok.c1.scan import BUDGETS

from .evaluation import CLOSED
from .protocol import LIMITS, OUTPUT

mpl.use("Agg")
COLORS = ("#26748D", "#CB6133", "#698B49", "#565D68")


def new_axes():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.7, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    ax.grid(axis="y", color="#D8DADC", linewidth=0.55)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.svg")
    fig.savefig(folder / f"{name}.png", dpi=600)
    plt.close(fig)


def budget_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [f"{k:,}" for k in BUDGETS])
    ax.set_xlabel("Displayed FSQ symbols per frame")


def s1_curves(summary):
    fig, ax = new_axes()
    points = summary["validation"]["points"]
    for policy, color, marker in zip(("SP-1", "SP-2", "SP-3", "C1-A1"), COLORS, ("o", "s", "^", "D")):
        rows = sorted((r for r in points if (r["strategy"], r["method"], r["history"]) == (policy, "B2", CLOSED)),
                      key=lambda r: r["budget"])
        ax.plot([r["budget"] for r in rows], [r["all"]["psnr_global_mse"] for r in rows],
                marker=marker, markersize=4, linewidth=1.4, color=color, label=policy)
    budget_axis(ax)
    ax.set_ylabel("Global PSNR (dB)")
    ax.set_title("Spatial observations and residual refinement", loc="left", fontweight="bold")
    ax.legend(loc="lower right", ncols=2)
    save(fig, "s1_closed_rate_distortion")


def s2_gains(decisions):
    fig, ax = new_axes()
    ax.axhline(0, color="#565D68", linewidth=0.8)
    ax.axhline(LIMITS["focus_min_psnr_gain_db"], color="#565D68", linewidth=0.8, linestyle=":")
    for (name, record), color, marker in zip(decisions.items(), COLORS, ("o", "s")):
        for scope, label, style in (("validation", "174f", "-"), ("horizon", "64f", "--")):
            rows = [r for r in record["gains"][scope] if r["history"] == CLOSED]
            ax.plot([r["budget"] for r in rows], [r["psnr_gain_db"] for r in rows],
                    label=f"{name} · {label}", color=color, marker=marker, markersize=4, linestyle=style)
    budget_axis(ax)
    ax.set_ylabel("Independent B3 − B2 PSNR (dB)")
    ax.set_title("Closed-loop gains of sparse recovery", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.39), ncols=4)
    save(fig, "s2_closed_psnr_gain")


def gradient_cosines(diagnostics):
    fig, ax = new_axes()
    ax.axhline(0, color="#565D68", linewidth=0.8)
    pairs = ("mse/lpips", "mse/anchor", "lpips/anchor")
    labels = ("MSE / LPIPS", "MSE / anchor", "LPIPS / anchor")
    for pair, label, color in zip(pairs, labels, COLORS):
        for variant, marker, style in (("W1", "o", "-"), ("W2", "s", "--")):
            rows = diagnostics[variant]
            ax.plot([r["step"] for r in rows], [r["cosines"][pair] for r in rows], color=color,
                    linewidth=1.2, marker=marker, markersize=3, linestyle=style, label=f"{label} · {variant}")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Trunk-gradient cosine similarity")
    ax.set_title("Agreement between reconstruction objectives", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncols=3, fontsize=7)
    save(fig, "s2_gradient_cosines")
