"""Compact paper figures using the recorded native measurements."""

import os
import string
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from PIL import Image

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from scripts.hla_tok.compact.inputs import CONFIG, CONTINUOUS, OUTPUT
from scripts.hla_tok.continuous.inputs import run_arguments
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3.protocol import read_config

INK = "#253342"
GRAY = "#77808c"
BLUE = "#2b6eae"
TEAL = "#208375"
GOLD = "#bf7b21"
PURPLE = "#834698"
PALETTE = (BLUE, TEAL, GOLD, PURPLE, GRAY)
METRICS = (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"))
LABELS = {"global": "Global PCA", "rank": "Rank-aware PCA", "identity": "C0 BF16"}


def _style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
                         "axes.labelsize": 7, "axes.titlesize": 7.5, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRAY,
                         "legend.frameon": False, "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "lines.linewidth": 1.1, "lines.markersize": 3.5})


def _fig(cols: int, rows: int = 1, height: float = 2.35):
    figure, axes = plt.subplots(rows, cols, figsize=(7.08, height), layout="constrained", squeeze=False)
    return figure, axes.flatten()


def _tags(axes) -> None:
    for ax, letter in zip(axes, string.ascii_lowercase):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=8)


def _lines(axis, rows, x, y, color, label, marker="o", style="-", **options):
    points = sorted((float(r[x]), float(r[y])) for r in rows if r.get(y, "") != "")
    if points:
        a, b = zip(*points, strict=True)
        axis.plot(a, b, color=color, marker=marker, linestyle=style, label=label, **options)


def spectrum(output: Path):
    figure, (a, b) = _fig(2)
    rows = csv_at(output / "pca_spectrum.csv")
    global_rows = [r for r in rows if r["scope"] == "global"]
    grid = np.asarray([[float(r["explained_fraction"]) for r in rows
                        if r["scope"] == "rank" and int(r["native_rank"]) == rank] for rank in range(121)])
    x = np.arange(1, 33)
    lo, mid, hi = np.quantile(grid, [.1, .5, .9], axis=0)
    a.fill_between(x, lo, hi, color=TEAL, alpha=.16, linewidth=0, label="Rank P10–P90")
    a.plot(x, mid, color=TEAL, linestyle="--", label="Rank median")
    _lines(a, global_rows, "component", "explained_fraction", BLUE, "Global", marker="")
    a.set(xlabel="Principal component", ylabel="Explained variance fraction", yscale="log", xlim=(1, 32))
    a.legend()
    effective = csv_at(output / "rank_effective_dimension.csv")
    _lines(b, [r for r in effective if r["scope"] == "rank"], "native_rank", "effective_rank_entropy",
           TEAL, "Native Rank", marker="")
    threshold = next(float(r["effective_rank_entropy"]) for r in effective if r["scope"] == "global")
    b.axhline(threshold, color=BLUE, linestyle="--", label="Global")
    b.set(xlabel="Native Rank ID", ylabel="Entropy effective dimension", xlim=(0, 120))
    b.legend()
    _tags((a, b))
    return figure


def projection(output: Path, config: dict):
    figure, (a, b) = _fig(2)
    rows = [r for r in csv_at(output / "projection_only_rd.csv") if r["order"] == "O2"]
    maxi = [r for r in rows if int(r["budget"]) == 16320]
    baseline = next(float(r["global_psnr"]) for r in maxi if r["projection"] == "identity")
    a.axhline(baseline, color=GRAY, linestyle=":", label="C0 BF16")
    for kind, color in (("global", BLUE), ("rank", TEAL)):
        selected = [r | {"dimension": int(r["profile"].split("-")[0][3:])} for r in maxi if r["projection"] == kind]
        _lines(a, selected, "dimension", "global_psnr", color, LABELS[kind], marker="s" if kind == "rank" else "o")
    a.set(xlabel="Transmitted components", ylabel="K16320 global PSNR (dB)", xticks=sorted(config["projection"]["dimensions"]))
    a.legend()
    for d, color in zip((4, 8, 16, 24), PALETTE):
        selected = [r for r in rows if r["profile"] == f"PCA{d}-BF16" and r["projection"] == "rank"]
        _lines(b, selected, "budget", "global_psnr", color, f"Rank PCA {d}D")
    _lines(b, [r for r in rows if r["projection"] == "identity"], "budget", "global_psnr", GRAY,
           "C0 BF16", marker="s", style=":")
    b.set(xlabel="Received Tokens", ylabel="Global PSNR (dB)", xscale="log", xticks=[510, 2040, 8160, 16320])
    b.set_xticklabels(["510", "2040", "8160", "16320"])
    b.legend(ncols=2, loc="lower right")
    _tags((a, b))
    return figure


def ablation(output: Path):
    fig, (a, b) = _fig(2)
    projection_rows = [r for r in csv_at(output / "projection_maximum_rd.csv") if r["projection"] != "global"]
    scalar_rows = [r for r in csv_at(output / "scalar_quantization_rd.csv")
                   if r["order"] == "O2" and int(r["budget"]) == 16320]
    _lines(a, projection_rows, "total_bpp", "global_psnr", GRAY, "Projection only", style=":", marker="s")
    _lines(a, [r for r in scalar_rows if r["profile"].startswith("Q32")], "total_bpp",
           "global_psnr", GOLD, "Quantization only", style="--", marker="^")
    _lines(a, [r for r in scalar_rows if r["projection"] != "identity"], "total_bpp",
           "global_psnr", TEAL, "Projection + quantization", style="", marker="o")
    a.set(xlabel="Total wire bits per pixel (log)", ylabel="K16320 global PSNR (dB)", xscale="log")
    a.legend(loc="lower right")
    rows = csv_at(output / "equal_bits_ablation.csv")
    for name, color in zip(("Q32-4", "Q16-4", "Q8-4"), PALETTE):
        selected = [r for r in rows if r["higher_dimension"] == name]
        label = name + " / " + selected[0]["lower_dimension"]
        _lines(b, selected, "budget", "higher_dimension_gain_global_psnr", color, label)
    b.axhline(0, color=GRAY, lw=.7)
    b.set(xlabel="Received Tokens", ylabel="Higher-dimension PSNR gain (dB)", xscale="log",
          xticks=[510, 2040, 8160, 16320])
    b.set_xticklabels(["510", "2040", "8160", "16320"])
    b.legend(loc="best")
    _tags((a, b))
    return fig


def _has(output: Path, name: str) -> list[dict]:
    return csv_at(output / name) if (output / name).is_file() else []


def rate_distortion(output: Path, selection: dict, training: dict):
    fig, axes = _fig(3, height=2.52)
    c0 = [r for r in csv_at(CONTINUOUS / "wire_rate_distortion.csv") if r["order"] == "O2" and r["wire_dtype"] == "bf16"]
    c1 = [r for r in csv_at(output / "single_frame_references.csv") if r["method"] == "C1 Fixed13"]
    original = csv_at(output / "original_rd.csv")
    untrained = [r for r in csv_at(output / "scalar_quantization_rd.csv")
                 if (r["asset"], r["order"]) == (selection["main_asset"], selection["order"])]
    trained = _has(output, "training_final_rd.csv")
    paths = [(c0, "C0 BF16", GRAY, "s", ":"), (c1, "C1 F13W", GOLD, "D", ""),
             (original, "Original AdapTok", PURPLE, "^", "--"), (untrained, "Compact static", TEAL, "o", "-"),
             (trained, "Compact adapted", BLUE, "v", "-")]
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for rows, name, color, marker, style in paths:
            _lines(axis, rows, "total_bpp", metric, color, name, marker, style)
        axis.set(xlabel="Total bits per pixel (log)", ylabel=label, xscale="log")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncols=len(names), loc="outside lower center")
    _tags(axes)
    return fig


def bandwidth(output: Path):
    fig, axes = _fig(3, height=2.52)
    rows = csv_at(output / "bandwidth_baseline_comparison.csv")
    data = [dict(row, x=i) for i, row in enumerate(rows)]
    targets = [row["target_mbps"] for row in rows]
    methods = (("c0", "C0 BF16", GRAY, "s", ":"), ("c1_f13w", "C1 F13W", GOLD, "D", ""),
               ("original", "Original AdapTok", PURPLE, "^", "--"), ("best_compact", "Best Compact", TEAL, "o", "-"))
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for key, name, color, marker, style in methods:
            _lines(axis, data, "x", key + "_" + metric, color, name, marker, style)
        axis.set(xlabel="Cap (Mbps at 30 FPS)", ylabel=label, xticks=range(len(targets)), xticklabels=targets)
        for tick in axis.get_xticklabels():
            tick.set(rotation=40, rotation_mode="anchor", ha="right")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncols=len(names), loc="outside lower center")
    _tags(axes)
    return fig


def training_curves(output: Path, training: dict):
    rows = _has(output, "training_curves.csv")
    trials = sorted({r["asset"] for r in rows})
    fig, axes = _fig(max(1, len(trials)), height=2.44)
    if not trials:
        axes[0].text(.5, .5, f"Adapter curves unavailable: {training['status']}", ha="center", va="center")
        axes[0].set_axis_off()
        return fig
    keys = (("hard_wire", "-"), ("continuous_bypass", "--"))
    for axis, asset in zip(axes, trials, strict=True):
        trial = [r for r in rows if r["asset"] == asset]
        for budget, group, color in ((min(int(r["budget"]) for r in trial), "Lowest cap", TEAL), (16320, "K16320", BLUE)):
            for key, style in keys:
                data = [r for r in trial if r["path"] == key and int(r["budget"]) == budget]
                _lines(axis, data, "step", "global_psnr", color, group + (" wire" if key == "hard_wire" else " bypass"),
                       marker="o" if key == "hard_wire" else "s", style=style)
        axis.set(xlabel="Optimizer updates", ylabel="Global PSNR (dB)")
        axis.set_title(asset.split("_seed")[0].replace("_rank", "").replace("_global", ""))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncols=4, loc="outside lower center")
    _tags(axes)
    return fig


def latency(output: Path, main: str):
    fig, (a, b) = _fig(2, height=2.6)
    rows = [r for r in csv_at(output / "latency_breakdown.csv") if r["asset"] == main]
    labels = [("encoder", "Encoder"), ("tokenizer", "Tokenizer"), ("ordering", "Ordering"),
              ("projection", "Projection"), ("quantization", "Scalar encode"), ("gpu_to_cpu", "GPU → CPU"),
              ("pack", "Packing"), ("unpack", "Unpacking"), ("cpu_to_gpu", "CPU → GPU"),
              ("dequantization", "Scalar decode"), ("lift", "Lift"), ("recovery", "Recovery"), ("rgb_decoder", "RGB")]
    timings = {(row["compute_dtype"], int(row["budget"]), row["stage"]): float(row["mean_ms"]) for row in rows}
    terms = [label for _, label in labels]
    y = np.arange(len(terms))
    for i, (precision, color) in enumerate((("bf16", TEAL), ("fp32", GRAY))):
        a.barh(y + (.19 if i else -.19), [timings[precision, 16320, key] for key, _ in labels],
               height=.35, color=color, label=precision.upper())
    a.set(xlabel="Stage mean at K16320 (ms)", yticks=y, yticklabels=terms)
    a.invert_yaxis()
    a.legend(loc="lower right")
    actual = csv_at(output / "throughput_benchmark.csv")
    for p, color, marker in (("bf16", TEAL, "o"), ("fp32", GRAY, "s")):
        data = [r for r in actual if r["asset"] == main and r["compute_dtype"] == p]
        _lines(b, data, "budget", "p95_ms", color, p.upper(), marker)
    b.axhline(33.33, color=GOLD, linestyle="--", label="30 FPS deadline")
    b.set(xlabel="Received Tokens", ylabel="Serial P95 (ms)", ylim=(0, None))
    b.legend(loc="best")
    _tags((a, b))
    return fig


def image_plate(output: Path, config: dict):
    folder = output / "samples"
    x, y, w, h = config["sample"]["crop_xywh"]
    sample = [("source", "Source"), ("C0_BF16_K16320", "C0 BF16 · K16320"),
              ("C1_Fixed13", "C1 Fixed13 · K16320"), ("Original_AdapTok_K2048_per_tile", "Original AdapTok · K2048/tile"),
              ("C0_BF16_5Mbps", "C0 BF16 · 5 Mbps cap"), ("Compact_5Mbps", "Compact · 5 Mbps cap"),
              ("Compact_EqualPayload", "Compact · C1 payload cap"), ("Compact_K16320", "Compact · K16320")]
    figure, axes = _fig(4, 4, height=4.6)
    index = []
    for i, (filename, label) in enumerate(sample):
        row = 8 * (i // 4) + i % 4
        source = folder / f"{filename}.png"
        with Image.open(source) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"sample lost native resolution: {source}")
            cropped = image.crop((x, y, x + w, y + h))
            crop_file = folder / f"{filename}_crop.png"
            cropped.save(crop_file)
            axes[row].imshow(image, interpolation="none")
            axes[row + 4].imshow(cropped, interpolation="none")
        axes[row].set_title(label)
        axes[row].set_axis_off()
        axes[row + 4].set_axis_off()
        index.append({"filename": source.name, "crop_filename": crop_file.name, "sha256": sha256_file(source),
                      "crop_sha256": sha256_file(crop_file), "crop_xywh": [x, y, w, h]})
    _tags(axes[::4])
    write_json(folder / "manifest.json", {"video": config["sample"]["video"], "frame": config["sample"]["frame"],
                                          "images": index, "native_size_wh": [1920, 1080]})
    return figure


def _export(fig, folder: Path, name: str, tools: Path | None) -> None:
    for axis in fig.axes:
        for scale, ticks in ((axis.get_xscale(), axis.xaxis), (axis.get_yscale(), axis.yaxis)):
            if scale == "log" and not isinstance(ticks.get_major_formatter(), matplotlib.ticker.FixedFormatter):
                ticks.set_major_formatter(StrMethodFormatter("{x:g}"))
    if tools is not None and len(fig.axes) > 1:
        sys.path.insert(0, str(tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(fig, json_out=folder / f"{name}.alignment.json")
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.svg")
    fig.savefig(folder / f"{name}.png", dpi=300)
    plt.close(fig)


def run(output: Path, tools: Path | None = None) -> None:
    _style()
    config = read_config(CONFIG)
    selection = json_at(output / "no_training_selection.json")
    training = json_at(output / "training_selection.json")
    main = training["asset"] if training.get("adopted") else selection["main_asset"]
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = (("figure_1_spectrum", lambda: spectrum(output)), ("figure_2_projection", lambda: projection(output, config)),
            ("figure_3_ablation", lambda: ablation(output)), ("figure_4_rate_distortion", lambda: rate_distortion(output, selection, training)),
            ("figure_5_bandwidth", lambda: bandwidth(output)), ("figure_6_training", lambda: training_curves(output, training)),
            ("figure_7_latency", lambda: latency(output, main)), ("figure_8_samples", lambda: image_plate(output, config)))
    for name, draw in jobs:
        _export(draw(), folder, name, tools)
    if tools is not None:
        _audit(folder, tools)


def _audit(folder: Path, tools: Path) -> None:
    subprocess.run([sys.executable, str(tools / "validate_figure.py"), __file__], check=True)
    for file in sorted(folder.glob("*.pdf")):
        with file.with_suffix(".text.json").open("w") as out:
            subprocess.run([sys.executable, str(tools / "audit_pdf_text.py"), str(file), "--json"],
                           stdout=out, check=True)
        subprocess.run([sys.executable, str(tools / "audit_figure_collisions.py"), str(file),
                        "--json-out", str(file.with_suffix(".collision.json"))], check=True)


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    configured = os.environ.get("NATURE_FIGURE_SCRIPTS")
    args = parser.parse_args()
    run(args.output, Path(configured) if configured else None)


if __name__ == "__main__":
    main()
