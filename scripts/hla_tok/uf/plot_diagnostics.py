"""Make diagnostic figures from frozen UF measurements and manifested samples."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image

from hla_tok.uf.official import sha256_file
from scripts.hla_tok.uf.evidence import compare
from scripts.hla_tok.v2.protocol import csv_at, json_at

COLORS = ("#355c7d", "#8f92a6", "#00887d", "#a2623d", "#b64c6d", "#574b7a")
BUDGETS = (5, 10, 20, 40)


def _axis() -> tuple[Figure, Axes]:
    return plt.subplots(figsize=(6.15, 3.65), layout="constrained")


def _save(figure: Figure, path: Path, raster: bool = False) -> None:
    for extension in ((".png",) if raster else (".pdf", ".svg", ".png")):
        figure.savefig(path.with_suffix(extension), dpi=600)
    plt.close(figure)


def _legend(axis: Axes, columns: int = 2) -> None:
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=columns, frameon=False, borderaxespad=0)


def _log_rate(axis: Axes, rates: Iterable[float]) -> None:
    if any(not 0 < rate for rate in rates):
        raise ValueError("log axis requires strictly positive rates")
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda rate, _: f"{rate:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())


def require_matplotlib_panel_alignment(figure: Figure, panels: list[list[Axes]]) -> None:
    figure.canvas.draw()
    coordinates = [[axis.get_position() for axis in row] for row in panels]
    for row in coordinates:
        for field in ("y0", "y1", "width"):
            values = [getattr(box, field) for box in row]
            extent = figure.get_figwidth() if field == "width" else figure.get_figheight()
            if (max(values) - min(values)) * extent * 72 > 1.5:
                raise ValueError(f"rendered panel row is misaligned: {field}")
    for column in zip(*coordinates, strict=True):
        for field in ("x0", "x1"):
            values = [getattr(box, field) for box in column]
            if (max(values) - min(values)) * figure.get_figwidth() * 72 > 1.5:
                raise ValueError(f"rendered panel column is misaligned: {field}")


def architecture(path: Path) -> None:
    figure, axis = _axis()
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    nodes = {
        "Frame t": (0.08, 0.69), "Native LD\nencoder": (0.28, 0.69), "Base y + z\nshared rANS": (0.52, 0.81),
        "Anchor RGB / D0 latent\nrefinement rANS": (0.52, 0.51), "HUF1 prefix\nreceiver": (0.76, 0.69),
        "Display t": (0.93, 0.69), "Base-only state\nfor frame t+1": (0.76, 0.22),
    }
    links = (("Frame t", "Native LD\nencoder"), ("Native LD\nencoder", "Base y + z\nshared rANS"),
             ("Native LD\nencoder", "Anchor RGB / D0 latent\nrefinement rANS"),
             ("Base y + z\nshared rANS", "HUF1 prefix\nreceiver"),
             ("Anchor RGB / D0 latent\nrefinement rANS", "HUF1 prefix\nreceiver"),
             ("HUF1 prefix\nreceiver", "Display t"), ("HUF1 prefix\nreceiver", "Base-only state\nfor frame t+1"))
    boxes = {label: axis.text(*position, label, ha="center", va="center",
                             bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f7f8", "edgecolor": COLORS[0]})
             for label, position in nodes.items()}
    figure.canvas.draw()
    for source, target in links:
        axis.annotate("", xy=nodes[target], xytext=nodes[source], arrowprops={
            "arrowstyle": "->", "color": COLORS[0], "shrinkA": 2, "shrinkB": 2,
            "patchA": boxes[source].get_bbox_patch(), "patchB": boxes[target].get_bbox_patch(),
        })
    axis.text(0.52, 0.08, "Enhancement never writes the recurrent native state", ha="center", color=COLORS[2])
    _save(figure, path)


def equal_quality(rows: list[dict], path: Path) -> None:
    points = [row for row in rows if row["metric"] == "lpips" and row["status"] == "PASS"]
    figure, axis = _axis()
    rates = []
    for key, label, color in (("reference_mbps", "Original AdapTok", COLORS[0]),
                              ("candidate_mbps", "Anchor A2 interpolated", COLORS[2])):
        measured = [float(row[key]) for row in points]
        rates.extend(measured)
        axis.plot(measured, [row["target_quality"] for row in points], "o-", label=label, color=color)
    _log_rate(axis, rates)
    axis.set(xlabel="Actual application rate (Mbps)", ylabel="Matched LPIPS-Alex")
    _legend(axis)
    _save(figure, path)


def stack_budgets(rows: list[dict], parts: tuple[tuple[str, str], ...], ylabel: str, scale: float, path: Path) -> None:
    figure, axis = _axis()
    bottom = [0.0] * len(BUDGETS)
    for (key, label), color in zip(parts, COLORS, strict=False):
        values = [mean(float(row[key]) for row in rows if int(row["budget_mbps"]) == budget) / scale
                  for budget in BUDGETS]
        axis.bar(range(len(BUDGETS)), values, bottom=bottom, label=label, color=color, width=0.62)
        bottom = [old + value for old, value in zip(bottom, values, strict=True)]
    axis.set(xticks=range(len(BUDGETS)), xticklabels=BUDGETS, xlabel="Nominal cap (Mbps)", ylabel=ylabel)
    _legend(axis, 3)
    _save(figure, path)


def images(dynamic: Path, output: Path) -> None:
    entries = json_at(dynamic / "samples/manifest.json")["samples"]
    sample = next(entry for entry in entries if "native_images" in entry)
    panels: list[Image.Image] = []
    for entry in sample["native_images"]:
        path = dynamic / "native" / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"image hash mismatch: {path}")
        with Image.open(path) as loaded:
            if loaded.size != (1920, 1080):
                raise ValueError(f"non-native image: {path}")
            panels.append(loaded.copy())
    labels = ("Source", "Base only", "5 Mbps", "10 Mbps", "20 Mbps", "40 Mbps")
    for name, crop in (("native_preview", None), ("native_crop", (768, 400, 1152, 616))):
        figure, axes = plt.subplots(2, 3, figsize=(7.15, 3.12), layout="constrained")
        for image, label, axis in zip(panels, labels, axes.flat, strict=True):
            shown = image.resize((600, 338)) if crop is None else image.crop(crop)
            axis.imshow(shown, interpolation="nearest")
            axis.set_title(label, loc="left", pad=3)
            axis.axis("off")
        require_matplotlib_panel_alignment(figure, [list(row) for row in axes])
        _save(figure, output / name, raster=True)


def dynamic_quality(rows: list[dict], path: Path) -> None:
    video = rows[0]["video"]
    selected = [row for row in rows if row["video"] == video]
    frames = [int(row["frame"]) for row in selected]
    figure, (quality, cap) = plt.subplots(2, 1, figsize=(6.15, 4.6), sharex=True, layout="constrained",
                                          gridspec_kw={"height_ratios": (2, 1)})
    for key, label, color in (("base_psnr", "Base only", COLORS[0]), ("display_psnr", "Received prefix", COLORS[2])):
        quality.plot(frames, [float(row[key]) for row in selected], label=label, color=color, linewidth=1)
    quality.set_ylabel("Frame PSNR (dB)")
    _legend(quality)
    cap.step(frames, [int(row["budget_mbps"]) for row in selected], where="post", color=COLORS[3], linewidth=0.8)
    cap.set(xlabel=f"Frame in {video.split(':')[-1]}", ylabel="Cap (Mbps)", yticks=BUDGETS)
    require_matplotlib_panel_alignment(figure, [[quality], [cap]])
    _save(figure, path)


def long_state(rows: list[dict], path: Path) -> None:
    figure, axis = _axis()
    for video, color in zip(dict.fromkeys(row["video"] for row in rows), COLORS, strict=False):
        selected = [row for row in rows if row["video"] == video]
        frames = [int(row["frame"]) for row in selected]
        label = video.split(":")[-1]
        mismatches = sum(row["state_exact"] != "True" for row in selected)
        axis.plot(frames, [int(row["base_bytes"]) for row in selected], color=color, linewidth=0.9,
                  label=f"{label} ({mismatches}/{len(frames)} state mismatches)")
    axis.axhline(9212, linestyle="--", color=COLORS[3], linewidth=1, label="Base hard cap")
    axis.set(xlabel="Native closed-loop frame", ylabel="Base bytes per frame")
    _legend(axis)
    _save(figure, path)


def training(path: Path, output: Path) -> None:
    rows = csv_at(path)
    figure, axis = _axis()
    groups = [rows[start:start + 25] for start in range(0, len(rows), 25)]
    for budget, color in zip(BUDGETS, COLORS, strict=False):
        axis.plot([int(group[-1]["step"]) for group in groups],
                  [mean(float(row[f"distortion_{budget}"]) for row in group) for group in groups],
                  label=f"{budget} Mbps", color=color, linewidth=1)
    axis.axvline(500, linestyle="--", linewidth=1, color=COLORS[4], label="A1 → A2 resume")
    axis.set(xlabel="Optimizer updates (25-step means)", ylabel="Train crop distortion objective")
    _legend(axis, 3)
    _save(figure, output)


def display_ablation(entries: dict[str, list[dict]], output: Path) -> None:
    figure, axis = _axis()
    rates = []
    for (name, rows), color in zip(entries.items(), COLORS, strict=False):
        points = [row for row in rows if row["budget_mbps"] in BUDGETS]
        measured = [float(row["actual_mbps"]) for row in points]
        rates.extend(measured)
        axis.plot(measured, [row["lpips"] for row in points], "o-", color=color, label=name, markersize=4, linewidth=1)
    _log_rate(axis, rates)
    axis.set(xlabel="Actual application rate (Mbps)", ylabel="LPIPS-Alex (untrained D0)")
    _legend(axis)
    _save(figure, output)


def native_penalty(entries: dict[str, list[dict]], reference: list[dict], path: Path) -> None:
    figure, axis = _axis()
    for (name, rows), color in zip(entries.items(), COLORS, strict=False):
        points = [row for row in compare(rows, reference) if row["metric"] == "global_psnr"
                  and row["budget_mbps"] in BUDGETS and row["same_rate_status"] == "PASS"]
        axis.plot([row["candidate_actual_mbps"] for row in points], [row["same_rate_quality_gain"] for row in points],
                  "o-", label=name, color=color, markersize=4, linewidth=1)
    axis.axhline(0, color="#808080", linewidth=0.8)
    axis.set(xlabel="Actual application rate (Mbps), overlap only", ylabel="PSNR minus official LD (dB)")
    _legend(axis)
    _save(figure, path)


def render(root: Path, output: Path) -> None:
    out_anchor = root / "anchor/A2-validation-bf16"
    anchor, frame_rows = json_at(out_anchor / "summary.json")["rows"], csv_at(out_anchor / "per_frame.csv")
    variants = {label: json_at(root / f"progressive/D0-{key}/summary.json")["rows"] for label, key in
                (("Ranked + conditional", "validation"), ("Fixed scan", "no-ranking"), ("Static CDF", "static-cdf"))}
    comparison = json_at(root / "evidence/A2_comparison.json")
    out_dynamic = root / "benchmarks/dynamic-A2"
    frames = csv_at(out_dynamic / "frames.csv")
    architecture(output / "architecture")
    equal_quality(comparison["at_reference_qualities"]["Original_AdapTok"], output / "equal_lpips")
    stack_budgets(frame_rows, (("base_codec_bytes", "Base rANS (y + z)"), ("base_header_bytes", "Base header"),
                               ("enhancement_codec_bytes", "Enhancement rANS"), ("enhancement_headers", "Record headers")),
                  "Mean application kB per frame", 1000, output / "application_bytes")
    images(out_dynamic, output)
    stack_budgets(frame_rows, (("native_encode_ms", "Native encode"), ("analysis_base_decode_ms", "Base preview"),
                               ("analysis_ms", "RGB analysis"), ("quant_rank_entropy_pack_ms", "Quantize/rank/rANS/pack"),
                               ("parse_decode_update_ms", "Parse/decode/update"), ("display_ms", "Display")),
                  "Mean serial time per frame (ms)", 1, output / "latency_components")
    dynamic_quality(frames, output / "dynamic_quality")
    long_state(frames, output / "long_base_state")
    training(root / "anchor/isolated-training-seed1/training.csv", output / "anchor_training")
    display_ablation(variants, output / "d0_ablation")
    native_penalty({"Anchor A2": anchor, "Unified D0": variants["Ranked + conditional"]},
                   comparison["archive_curves"]["B3_gop32"], output / "ld_psnr_penalty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}):
        render(args.results, args.output)


if __name__ == "__main__":
    main()
