"""Plot measured 174-frame Validation curves without extrapolation."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from scripts.hla_tok.v2.protocol import json_at

STYLE = {
    "Original_AdapTok": ("Original AdapTok", "#5f6975", "v", ":"),
    "T0_RA_S2": ("HLA-Tok-T S2", "#8c6d38", "s", "--"),
    "B3_gop32": ("Official LD (GOP 32)", "#4c708f", "o", "-"),
    "B4": ("Official Intra", "#a0a5ad", "d", "--"),
    "Anchor A2": ("Anchor A2 (BF16)", "#007f79", "^", "-"),
    "Unified D0": ("Unified D0 (untrained)", "#a04271", "P", "-"),
}


def plot(curves: dict, metric: str, ylabel: str, target: Path) -> None:
    with mpl.rc_context({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(6.25, 3.7), layout="constrained")
        for method, (label, color, marker, line) in STYLE.items():
            rows = sorted(curves.get(method, []), key=lambda row: row["actual_mbps"])
            if rows:
                if any(not 0 < row["actual_mbps"] for row in rows):
                    raise ValueError(f"{method} contains a non-positive actual rate")
                axis.plot([row["actual_mbps"] for row in rows], [row[metric] for row in rows],
                          label=label, color=color, marker=marker, linestyle=line, markersize=4, linewidth=1.1)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda rate, _: f"{rate:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Actual application data rate at 30 FPS (Mbps)")
        axis.set_ylabel(ylabel)
        axis.legend(loc="lower left" if metric == "lpips" else "upper left", ncol=2, frameon=False,
                    bbox_to_anchor=(0, 1.01), borderaxespad=0)
        for extension in (".pdf", ".svg", ".png"):
            figure.savefig(target.with_suffix(extension), dpi=600)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--unified", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves = json_at(args.comparison)["archive_curves"]
    for name, source in (("Anchor A2", args.anchor), ("Unified D0", args.unified)):
        if source is not None:
            curves[name] = [row for row in json_at(source)["rows"] if row["status"] == "PASS"]
    args.output.mkdir(parents=True, exist_ok=True)
    plot(curves, "global_psnr", "Global RGB PSNR (dB)", args.output / "validation_psnr")
    plot(curves, "lpips", "LPIPS-Alex (lower is better)", args.output / "validation_lpips")
    plot(curves, "dists", "DISTS (lower is better)", args.output / "validation_dists")


if __name__ == "__main__":
    main()
