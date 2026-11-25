"""Publication figures from measured HLT tables and recorded native reconstructions."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from PIL import Image

from scripts.hla_tok.compact.figures import BLUE, GOLD, GRAY, INK, PURPLE, TEAL, _lines, _style, _tags
from scripts.hla_tok.temporal.inputs import COMPACT, RESULTS
from scripts.hla_tok.v2.protocol import csv_at, json_at

BUDGETS = ("5", "10", "20", "40")
METRICS = (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"))
DIAGNOSTICS = ("A0 absolute", "A1 oracle", "A2 CP decoded", "A3 copy CLOSED", "A4 linear CLOSED")
METHOD_COLORS = {"Original AdapTok": ("#333333", "o", ":"),
                 "CP Q16-4": (GRAY, "s", "--"), "CP Q8-4 T1": (PURPLE, "^", "--"),
                 "C1 Fixed13": (GOLD, "D", ":"), "Intra RA": (BLUE, "s", "-"),
                 "T0 fixed": ("#5a899e", "v", "--"), "T0-RA": (TEAL, "o", "-")}


def _grid(rows: int, columns: int, height: float = 2.6, *, bottom: float = .18, right: float = .97):
    fig, axes = plt.subplots(rows, columns, figsize=(7.08, height), squeeze=False)
    fig.subplots_adjust(left=.105, right=right, top=.89, bottom=bottom, wspace=.46, hspace=.58)
    return fig, axes.ravel()


def architecture(_: Path):
    fig, ax = plt.subplots(figsize=(7.08, 2.32))
    fig.subplots_adjust(left=.035, right=.975, top=.94, bottom=.075)
    ax.set(xlim=(-1, 100), ylim=(0, 38))
    ax.set_axis_off()
    descriptions = [(0, 27, "Native 1080p\nRGB frame"), (17, 27, "C0 spatial\nencoder + AQT"),
                    (34, 27, "Rank PCA\n32D lattice"), (51, 27, "Causal\ninnovation"),
                    (68, 27, "HLT records\n+ CRC"), (85, 27, "Actual bytes\nbase + enhance"),
                    (85, 5, "Independent\nreceiver"), (68, 5, "Integer\nrecovery"),
                    (51, 5, "Display\nprefix"), (34, 5, "Rank lift\n+ Local D0"),
                    (17, 5, "Native 1080p\nreconstruction")]
    for x, y, label in descriptions:
        color = BLUE if y == 27 else TEAL
        node = FancyBboxPatch((x, y), 14, 7.2, boxstyle="round,pad=.3,rounding_size=1", ec=color,
                              fc="white", linewidth=.85)
        ax.add_patch(node)
        ax.text(x + 7, y + 3.6, label, ha="center", va="center", fontsize=7)
    for left in (0, 17, 34, 51, 68):
        ax.add_patch(FancyArrowPatch((left + 14.6, 30.6), (left + 16.65, 30.6),
                                    arrowstyle="-|>", lw=.8, color=INK, mutation_scale=7))
    for right in (85, 68, 51, 34):
        ax.add_patch(FancyArrowPatch((right - .3, 8.6), (right - 2.65, 8.6),
                                    arrowstyle="-|>", lw=.8, color=INK, mutation_scale=7))
    ax.annotate("", xy=(92, 12.5), xytext=(92, 26.5), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": .8})
    ax.add_patch(FancyBboxPatch((34, 16.6), 18, 5.4, boxstyle="round,pad=.3,rounding_size=1",
                              ec=GOLD, fc="white", linewidth=.9))
    ax.text(43, 19.2, "Base at frame t − 1", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((52.5, 20.9), (56.5, 26.6), arrowstyle="-|>", color=GOLD,
                                 lw=.9, mutation_scale=7))
    ax.add_patch(FancyBboxPatch((58, 16.6), 25, 5.4, boxstyle="round,pad=.3,rounding_size=1",
                              ec=GOLD, fc="white", linewidth=.9))
    ax.text(70.5, 19.2, "Commit complete base at frame t", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((75, 12.5), (75, 16.1), arrowstyle="-|>", color=GOLD, lw=.9, mutation_scale=7))
    ax.text(0, 35.8, "Sender", color=BLUE, fontweight="bold")
    ax.text(0, 7, "Receiver", color=TEAL, fontweight="bold")
    return fig


def correlation(output: Path):
    rows = csv_at(output / "rank_temporal_correlation.csv")
    fig, axes = _grid(1, 1, 1.88, bottom=.27, right=.87)
    axis = axes[0]
    names = DIAGNOSTICS[1:]
    grid = [[float(row["pearson"]) for row in rows if row["method"] == name] for name in names]
    panel = axis.imshow(grid, aspect="auto", vmin=.45, vmax=.80, cmap="cividis", interpolation="nearest")
    axis.set(yticks=list(range(len(names))), yticklabels=names, xlabel="Native rank",
             xticks=[0, 30, 60, 90, 120])
    axis.tick_params(axis="y", length=0)
    bar = fig.add_axes([.9, .27, .015, .62], label="<colorbar>")
    fig.colorbar(panel, cax=bar, label="Pearson r")
    fig.subplots_adjust(left=.18)
    return fig


def energy(output: Path):
    rows = [r for r in csv_at(output / "residual_spectrum.csv") if r["eigen_index"] == "0"]
    fig, axes = _grid(1, 2, bottom=.28)
    for label, color, method in [("Absolute", GRAY, DIAGNOSTICS[0]), ("Oracle innovation", BLUE, DIAGNOSTICS[1]),
                                 ("CLOSED innovation", TEAL, DIAGNOSTICS[3])]:
        data = [r for r in rows if r["method"] == method]
        for axis, metric in zip(axes, ("trace", "effective_dimension"), strict=True):
            _lines(axis, data, "rank", metric, color, label, marker="")
    axes[0].set(xlabel="Native rank", ylabel="Centered covariance trace", yscale="log", yticks=[3, 5, 10, 20])
    axes[1].set(xlabel="Native rank", ylabel="Entropy effective dimension")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.53, .01))
    _tags(axes)
    return fig


def rate_distortion(output: Path):
    chosen = json_at(output / "candidate_selection.json")["method"]
    rows = csv_at(output / "bandwidth_comparison.csv")
    rows = [r for r in rows if r["bandwidth"] in BUDGETS and r["method"] != "Original AdapTok"]
    rows += csv_at(COMPACT / "original_rd.csv")
    seen = {}
    for r in rows:
        seen[(r["method"], r["mbps_30fps"], r["global_psnr"])] = r
    colors = METHOD_COLORS | {chosen: (TEAL, "o", "-")}
    fig, axes = _grid(1, 3, 2.35, bottom=.32)
    methods = list(dict.fromkeys(("Original AdapTok", "C1 Fixed13", "CP Q16-4", "CP Q8-4 T1",
                                  "Intra RA", "T0 fixed", chosen)))
    for ax, (metric, label) in zip(axes, METRICS, strict=True):
        for m in methods:
            color, marker, style = colors[m]
            data = [r for r in seen.values() if r["method"] == m]
            _lines(ax, data, "mbps_30fps", metric, color, m, marker, style, linewidth=1.5 if m == chosen else .95)
        ax.set(xlabel="Application rate (Mbps)", ylabel=label, xlim=(0, 42), xticks=[0, 10, 20, 30, 40])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(.53, .012))
    _tags(axes)
    return fig


def allocation(output: Path):
    chosen = json_at(output / "candidate_selection.json")["method"]
    rows = csv_at(output / "rank_component_allocation.csv")
    names = {r["method"] for r in rows}
    method = chosen if chosen in names else "T0-RA"
    fig, axes = _grid(1, 4, 2.14, bottom=.24, right=.88)
    for index, (ax, bandwidth) in enumerate(zip(axes, BUDGETS, strict=True)):
        values = np.zeros((121, 32))
        for row in rows:
            if row["method"] == method and row["bandwidth"] == bandwidth:
                values[int(row["rank"]), int(row["component"])] = float(row["fraction_of_actual_regions"])
        panel = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap="cividis", interpolation="nearest",
                          origin="lower")
        ax.set(xlabel="Component", title=f"{bandwidth} Mbps", xticks=(0, 15, 31), yticks=(0, 60, 120))
        if index:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Native rank")
    bar = fig.add_axes([.90, .24, .014, .65], label="<colorbar>")
    fig.colorbar(panel, cax=bar, label="Fraction of actual regions")
    return fig


def _soapbox(output, condition):
    return [r for r in csv_at(output / "closed_loop_per_frame.csv")
            if r["video"] == "DAVIS2017:soapbox" and r["condition"] == condition]


def long_loop(output: Path):
    rows = _soapbox(output, "fixed_10")
    fig, axes = _grid(2, 2, 3.9, bottom=.20)
    for ax, (metric, label) in zip(axes[:3], (("psnr", "Frame PSNR (dB)"), *METRICS[1:]), strict=True):
        _lines(ax, rows, "frame", metric, TEAL, label, marker="")
        ax.set(ylabel=label, xlim=(0, 98))
    for field, color in (("prediction_rms", BLUE), ("residual_rms", GOLD), ("reconstruction_rms", TEAL)):
        _lines(axes[3], rows, "frame", field, color, field.removesuffix("_rms").title(), marker="")
    axes[3].set(ylabel="Coefficient RMS", xlim=(0, 98))
    fig.legend(*axes[3].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.53, .015))
    for ax in axes[2:]:
        ax.set_xlabel("Source frame")
    for ax in axes:
        for row in rows:
            if row["frame_type"] == "I":
                ax.axvline(int(row["frame"]), color=GRAY, lw=.6, alpha=.45)
    _tags(axes)
    return fig


def dynamic_bandwidth(output: Path):
    rows = _soapbox(output, "5-40-5")
    fig, axes = _grid(2, 1, 3.6, bottom=.15)
    axes[0].step([int(r["frame"]) for r in rows], [float(r["cap_bytes"]) / 1000 for r in rows],
                 where="mid", label="Budget", color=GRAY, linestyle="--")
    with_bytes = [r | {"kb": float(r["actual_bytes"]) / 1000} for r in rows]
    _lines(axes[0], with_bytes, "frame", "kb", TEAL, "Application wire", marker="")
    _lines(axes[1], rows, "frame", "psnr", BLUE, "PSNR", marker="")
    axes[0].set(ylabel="kB per frame", xlim=(0, 98))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.53, .99))
    axes[1].set(xlabel="Source frame", ylabel="Frame PSNR (dB)", xlim=(0, 98))
    _tags(axes)
    return fig


def digests(output: Path):
    modes = {"fixed_40": "Full", "5-40-5": "5 → 40 → 5 Mbps", "periodic_enhancement_loss": "Periodic enhancement loss",
             "all_enhancement_lost": "All enhancements lost", "base_loss_wait_I": "Base record lost"}
    def status(row: dict) -> int:
        if row["quality_status"] == "WAIT_I":
            return 0
        if row["encoder_reference_digest"] != row["receiver_reference_digest"]:
            return 3
        return 2 if row["event"] == "I_recovery" else 1
    states = [[status(row) for row in _soapbox(output, mode)] for mode in modes]
    fig, axes = _grid(1, 1, 1.95, bottom=.34)
    fig.subplots_adjust(left=.29)
    ax = axes[0]
    colors = [GOLD, TEAL, BLUE, "#b94442"]
    ax.imshow(states, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors), vmin=0, vmax=3)
    ax.set(xlabel="Source frame", yticks=range(len(modes)), yticklabels=list(modes.values()))
    ax.tick_params(axis="y", length=0)
    markers = [Line2D([], [], marker="s", ls="", color=color, label=label) for color, label in zip(
        colors, ("Waiting for I", "Digests match", "I recovery", "Mismatch"), strict=True)]
    fig.legend(handles=markers, loc="lower center", ncol=4, bbox_to_anchor=(.55, .01))
    return fig


def bytes_used(output: Path):
    method = json_at(output / "candidate_selection.json")["method"]
    data = csv_at(output / "rate_breakdown.csv")
    rows = [next(r for r in data if r["method"] == method and r["bandwidth"] == b and r["basis"] == "total_wire")
            for b in BUDGETS]
    fig, axes = _grid(1, 1, 2.3, bottom=.23)
    ax = axes[0]
    marks = np.arange(len(rows))
    bitmap = np.asarray([float(r["bitmap_bytes"]) for r in rows])
    base = np.asarray([float(r["base_payload_bytes"]) for r in rows])
    enhance = np.asarray([float(r["enhancement_payload_bytes"]) for r in rows])
    other = [base + enhance - bitmap, bitmap,
             np.asarray([float(r["header_bytes"]) + float(r["record_header_bytes"]) for r in rows]),
             np.asarray([float(r["crc_bytes"]) for r in rows])]
    last = np.zeros_like(bitmap)
    for values, label, color in zip(other, ("Symbols", "Bitmap", "Headers", "CRC"), (TEAL, BLUE, GOLD, GRAY), strict=True):
        v: np.ndarray = values / 1000
        ax.bar(marks, v, bottom=last, width=.55, color=color, label=label)
        last += v
    ax.set(xticks=marks, xticklabels=BUDGETS, xlabel="Bandwidth budget (Mbps)", ylabel="Mean kB per frame")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(.53, .99))
    return fig


def latency(output: Path):
    table = csv_at(output / "latency_breakdown.csv")
    rows = [r for r in table if r["stage"] == "total_serial"]
    fig, axes = _grid(1, 2, 2.66, bottom=.28)
    for dtype, color in (("bf16", TEAL), ("fp32", BLUE)):
        _lines(axes[0], [r for r in rows if r["precision"] == dtype], "bandwidth", "p95_ms", color, dtype.upper())
    axes[0].axhline(33.33, color=GOLD, lw=.8, ls="--", label="P95 limit (33.33 ms)")
    axes[0].set(xlabel="Bandwidth budget (Mbps)", ylabel="Serial P95 (ms)", xticks=(5, 10, 20, 40))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.53, .015))
    spans = [("spatial_encoder", "Spatial encoder"), ("aqt", "AQT"), ("rank_transform", "PCA"),
             ("encode_application_wire", "Encode wire"), ("receive_application_wire", "Receive wire"),
             ("local_recovery_and_lift", "Lift and local decode"), ("d0_rgb", "D0 RGB")]
    values = [float(next(r["mean_ms"] for r in table if r["stage"] == item and r["precision"] == "bf16"
                         and r["bandwidth"] == "40")) for item, _ in spans]
    axes[1].barh(np.arange(len(spans)), values, color=[BLUE]*3 + [GOLD]*2 + [TEAL]*2)
    axes[1].set(yticks=np.arange(len(spans)), yticklabels=[label for _, label in spans],
                xlabel="Mean ms at 40 Mbps, BF16")
    axes[1].invert_yaxis()
    fig.subplots_adjust(wspace=.68)
    _tags(axes)
    return fig


def ablation(output: Path):
    rows = csv_at(output / "ablation_comparison.csv")
    fig, axes = _grid(1, 2, 2.52, bottom=.3)
    labels = ("Intra fixed", "Intra RA", "T0 fixed", "T0 linear", "T0-RA RAW", "T0-RA", "T0-RA S2")
    colors = (GRAY, BLUE, "#6395ad", PURPLE, GOLD, TEAL, "#1b4e49")
    for label, color in zip(labels, colors, strict=True):
        series = [r for r in rows if r["method"] == label and r["basis"] == "total_wire" and r["bandwidth"] in BUDGETS]
        for ax, (metric, y) in zip(axes, METRICS[:2], strict=True):
            _lines(ax, series, "bandwidth", metric, color, label, marker="o", style="--" if "Intra" in label else "-")
            ax.set(xlabel="Bandwidth budget (Mbps)", ylabel=y, xticks=(5, 10, 20, 40))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(.53, .013))
    _tags(axes)
    return fig


def sample_plate(output: Path):
    groups = [("blackswan-25", "Low motion", (440, 355, 600, 337)), ("car-roundabout-32", "Camera motion", (710, 425, 600, 337)),
              ("dog-48", "Fine texture", (640, 345, 600, 337)), ("soapbox-98", "Long sequence", (680, 360, 600, 337)),
              ("scene-boundary", "Source switch", (640, 355, 600, 337))]
    method = json_at(output / "candidate_selection.json")["method"]
    fig, axes = plt.subplots(len(groups), 3, figsize=(7.08, 6.56))
    fig.subplots_adjust(left=.05, right=.98, top=.88, bottom=.024, hspace=.1, wspace=.08)
    for row, (source, content, crop) in enumerate(groups):
        folder = output / "samples" / source
        if source == "soapbox-98":
            paths = ("source.png", f"{method.replace(' ', '_')}_fixed_5.png", f"{method.replace(' ', '_')}_fixed_40.png")
        elif source == "scene-boundary":
            paths = ("source.png", "previous_source.png", "reconstruction.png")
        else:
            temporal = method if (folder / f"{method.replace(' ', '_')}_5Mbps.png").is_file() else "T0-RA"
            paths = ("source.png", "Intra_RA_5Mbps.png", f"{temporal.replace(' ', '_')}_5Mbps.png")
        x, y, w, h = crop
        for col, filename in enumerate(paths):
            ax = axes[row, col]
            with Image.open(folder / filename) as image:
                if image.size != (1920, 1080):
                    raise ValueError("sample is not native RGB")
                ax.imshow(image if col == 0 else image.crop((x, y, x + w, y + h)), interpolation="nearest")
            if not col:
                ax.add_patch(Rectangle((x, y), w, h, ec="#f5ae42", fill=False, lw=1))
            ax.set_axis_off()
        axes[row, 0].text(.01, .015, content, transform=axes[row, 0].transAxes, color="white", fontsize=6,
                          bbox={"boxstyle": "square,pad=.18", "facecolor": "#222222", "alpha": .8, "lw": 0})
    for ax, name in zip(axes[0], ("Native source", "Control or lower-rate crop", "Temporal crop"), strict=True):
        ax.set_title(name)
    return fig


def _export(figure, folder: Path, name: str, tools: Path) -> None:
    sys.path.insert(0, str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    if len(axes) > 1:
        require_matplotlib_panel_alignment(figure, json_out=folder / f"{name}.alignment.json")
    figure.savefig(folder / f"{name}.pdf", dpi=600)
    figure.savefig(folder / f"{name}.svg", dpi=600)
    figure.savefig(folder / f"{name}.png", dpi=600)
    figure.savefig(folder / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)


def audit(folder: Path, tools: Path) -> None:
    subprocess.run([sys.executable, str(tools / "validate_figure.py"), __file__], check=True)
    for pdf in sorted(folder.glob("*.pdf")):
        with pdf.with_suffix(".text.json").open("w") as stream:
            subprocess.run([sys.executable, str(tools / "audit_pdf_text.py"), str(pdf), "--json"],
                           stdout=stream, check=True)
        subprocess.run([sys.executable, str(tools / "audit_figure_collisions.py"), str(pdf),
                        "--json-out", str(pdf.with_suffix(".collision.json"))], check=True)


def run(output: Path, tools: Path, figures: list[int] | None = None) -> None:
    _style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    drawings = (architecture, correlation, energy, rate_distortion, allocation, long_loop, dynamic_bandwidth,
                digests, bytes_used, latency, ablation, sample_plate)
    for i, draw in enumerate(drawings, 1):
        if figures is None or i in figures:
            _export(draw(output), folder, f"figure_{i:02d}_{draw.__name__}", tools)
    audit(folder, tools)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=int, nargs="+")
    parser.add_argument("--tools", type=Path, default=os.environ.get("NATURE_FIGURE_SCRIPTS"))
    args = parser.parse_args()
    if args.tools is None:
        parser.error("--tools or NATURE_FIGURE_SCRIPTS is required")
    run(args.output, args.tools, args.figures)
