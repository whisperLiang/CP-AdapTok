"""Static diagnostic figures generated from the same machine-readable tables."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

from aip_vtok.runtime import file_hash, write_json

from .closed import FIX_A
from .fit import NATIVE_ORDER, ORDERS, RANKS
from .io import OUTPUT, PROBES
from .matched import CLEAN, LOW
from .packets import COARSE, FULL, HYBRID, LAYOUTS
from .report import PSNR, SPARSE_KEYS, pool, rank_rows, select

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 9, "axes.titlesize": 10, "svg.fonttype": "none", "svg.hashsalt": "aip-vtok-diag0",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "legend.frameon": False, "grid.alpha": 0.19,
}
COLORS = ("#366188", "#B25939", "#647469")
ORDER_LABELS = ("PCA variance", "PCA RGB sensitivity", "Native RGB sensitivity")
LABELS = {FULL: "Full-resolution depth", COARSE: "2 × 2 shared depth", HYBRID: "Hybrid", FIX_A: "Fix-A"}


def export(fig, output, name):
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ModuleNotFoundError:
        pass
    else:
        require_matplotlib_panel_alignment(fig, json_out=output / f"{name}.alignment.json", strict=True)
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def order_legend(fig, count):
    lines = [Line2D([], [], color=color, marker="o", markersize=3) for color in COLORS[:count]]
    fig.legend(lines, ORDER_LABELS[:count], loc="outside upper center", ncol=count)


def rank_figure(artifact, output):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.3), sharex=True, sharey="row", layout="constrained")
    scopes = ((CLEAN, "I", "R0 I"), (CLEAN, "P", "R0 P"), (LOW, "P", "Fix-A P510 history"))
    for col, (history, kind, title) in enumerate(scopes):
        summaries = rank_rows(artifact["summary"], history, kind)
        for color, order in zip(COLORS, (*ORDERS, NATIVE_ORDER)):
            rows = summaries[order]
            axes[0, col].plot(RANKS, [r[PSNR] for r in rows], "o-", color=color, markersize=2.7)
            axes[1, col].plot(RANKS, [100 * r["variance_explained_train"] for r in rows], "o-", color=color, markersize=2.7)
        axes[0, col].set_title(f"{title} · n={rows[0]['frames']}", loc="left")
        axes[1, col].set(xlabel="Retained rank", xticks=(6, 24, 48, 96, 128))
    axes[0, 0].set_ylabel("Global PSNR (dB)")
    axes[1, 0].set_ylabel("Train variance retained (%)")
    order_legend(fig, 3)
    for axis in axes.ravel():
        axis.grid(axis="y")
    export(fig, output, "rank")


def spatial_figure(artifact, output):
    cols = ("method", "block", "rank")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True, layout="constrained")
    for i, kind in enumerate(("I", "P")):
        rows = pool(select(artifact["rows"], history=CLEAN, frame_type=kind), cols)
        x = np.arange(len(rows)) + (i - 0.5) * 0.36
        axes[0].bar(x, [r[PSNR] for r in rows], width=0.35, color=COLORS[i], label=f"{kind} frames")
        axes[1].bar(x, [100 * r["latent_mse"] / r["latent_energy_zero"] for r in rows], width=0.35, color=COLORS[i])
    names = {
        "r0": "R0", "channel_pca": "1/16 channel",
        "bilinear": "1/{grid} bilinear", "block_pca": "1/{grid} block",
    }
    labels = []
    for row in rows:
        title = names[row["method"]].format(grid=16 * row["block"])
        labels.append(title if row["method"] == "r0" else f"{title}\n{row['rank']}D")
    axes[0].set_ylabel("Global PSNR (dB)")
    axes[1].set_ylabel("Residual / raw latent energy (%)")
    axes[1].set_xticks(np.arange(len(rows)), labels, fontsize=8)
    for axis in axes:
        axis.grid(axis="y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    export(fig, output, "spatial")


def draw_quality(axis, rows, xfield, yfield, versions):
    for index, order in enumerate(ORDERS):
        for version, style in versions:
            group = sorted(select(rows, order=order, version=version), key=lambda r: r["budget"])
            axis.plot(
                [r[xfield] / 1000 for r in group], [r[yfield] for r in group],
                marker="o", linestyle=style, markersize=3, color=COLORS[index],
            )
    axis.grid(axis="y")


def rate_legend(fig):
    handles = [Line2D([], [], color=c) for c in COLORS[:2]]
    handles.extend(Line2D([], [], color="#393939", ls=ls) for ls in ("--", "-", ":"))
    fig.legend(handles, (*ORDER_LABELS[:2], "Continuous", "FSQ", "DMC-LD actual bytes"),
               loc="outside upper center", ncol=5)


def rate_figure(art, native, output, *, depth):
    cols = len(LAYOUTS) if depth else 2
    fig, axes = plt.subplots(2, cols, figsize=(10.4, 5.6), sharey="row", layout="constrained")
    if depth:
        pool_rows = pool(select(art["rows"], history=CLEAN), SPARSE_KEYS)
        subsets = [(LABELS[layout], select(pool_rows, layout=layout)) for layout in LAYOUTS]
    else:
        subsets = [(f"Free Top-K · {kind}", select(art["summary"], history=CLEAN, cohort="all", frame_type=kind))
                   for kind in ("I", "P")]
    versions = (("continuous", "--"), ("fsq_quantized", "-"))
    baseline = native["bpp"] * 1920 * 1080 / 8000
    for col, (title, rows) in enumerate(subsets):
        top, bottom = axes[:, col]
        draw_quality(top, rows, "R_total_raw", PSNR, versions)
        if depth:
            draw_quality(bottom, rows, "R_total_entropy_ref", "lpips", versions[1:])
        else:
            draw_quality(bottom, rows, "R_total_raw", "lpips", versions)
        top.set(title=title, xlabel="Nominal raw size (kB/frame)")
        bottom.set_xlabel("Entropy + zlib reference (kB/frame)" if depth else "Nominal raw size (kB/frame)")
        for axis in (top, bottom):
            axis.axvline(baseline, color="#4c4c4c", linewidth=0.9, linestyle=":")
    axes[0, 0].set_ylabel("Global PSNR (dB)")
    axes[1, 0].set_ylabel("Full-frame LPIPS-Alex")
    rate_legend(fig)
    export(fig, output, "depth_quality_bytes" if depth else "free_quality_bytes")


def closed_figure(artifact, output):
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.8), sharex=True, layout="constrained")
    for color, case in zip(("#647469", "#366188", "#B25939"), artifact["candidates"]):
        layout = case["layout"]
        for axis, budget in zip(axes[:2], (510, 8192)):
            rows = select(artifact["rows"], layout=layout, budget=budget)
            axis.plot([r["frame"] for r in rows], [r["psnr"] for r in rows], label=LABELS[layout], color=color)
        axes[2].plot([r["frame"] for r in rows], [r["reference_latent_rms"] for r in rows], color=color)
    axes[0].set_ylabel("K510 frame PSNR (dB)")
    axes[1].set_ylabel("K8192 frame PSNR (dB)")
    axes[2].set(ylabel="Committed latent RMS", xlabel="car-roundabout frame index", xticks=(0, 16, 31, 48, 63))
    for axis in axes:
        axis.axvline(32, ls=":", color="#5a5a5a", linewidth=0.9)
        axis.axvspan(48, 63, color="#d7dbde", alpha=0.38, zorder=0)
        axis.grid(axis="y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    export(fig, output, "closed_loop")


def crop_figure(artifact, frame, output):
    budgets = (510, 2048, 4096, 8192)
    labels = ("Original", *(f"K{budget}" for budget in budgets))
    directory = output / "crops"
    width, height = 256, 256
    header, gap = 32, 4
    canvas = Image.new("RGB", (len(labels) * (width + gap), 3 * (height + header + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    original = directory / f"f{frame:03d}_original.png"
    for row, case in enumerate(artifact["candidates"]):
        layout = case["layout"]
        paths = [original, *(directory / f"f{frame:03d}_{layout}_{k}.png" for k in budgets)]
        for col, (label, path) in enumerate(zip(labels, paths)):
            x, y = col * (width + gap), row * (height + header + gap)
            draw.text((x + 5, y + 7), f"{label} · {LABELS[layout]}", fill="#212931", font=font)
            with Image.open(path) as image:
                canvas.paste(image, (x, y + header))
    canvas.save(output / f"crop_f{frame:03d}.png")


def build_figures(a, b, c, d, e, native):
    output = OUTPUT / "figures"
    with mpl.rc_context(STYLE):
        rank_figure(a, output)
        spatial_figure(b, output)
        rate_figure(c, native, output, depth=False)
        rate_figure(d, native, output, depth=True)
        closed_figure(e, output)
        for frame in (31, 63):
            crop_figure(e, frame, output)
    write_json(output / "sources.json", {
        "measured_inputs_sha256": {stem: file_hash(OUTPUT / f"{stem}.json") for stem in PROBES.values()},
        "crops_xyxy_1080p": e["crop_xyxy"],
        "crops": "unscaled 256 by 256 pixels, no contrast, gamma or color adjustment",
        "uncertainty": "descriptive matched-frame summaries of fixed clips, no inference or interval",
    })
