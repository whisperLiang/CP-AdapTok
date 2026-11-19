"""Static engineering figures from raw experiments; x-axis is tokens, not bpp."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from aip_vtok.evaluation.report_stage1 import available_results
from aip_vtok.evaluation.smoke import BUDGETS
from aip_vtok.runtime import file_hash, resolve, write_json


def training_plot(directory, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    paths = [directory / "gate1_smoke/training.json", *sorted(directory.glob("gate1_seed*/training.json"))]
    for index, path in enumerate(paths):
        history = json.loads(path.read_text())["history"]
        steps, loss = np.array([r["step"] for r in history]), np.array([r["loss"] for r in history])
        axis = axes[0 if index == 0 else 1]
        label = path.parent.name
        line = axis.plot(steps, loss, alpha=0.25, linewidth=0.6)[0]
        width = min(20, len(loss))
        axis.plot(
            steps[width - 1 :],
            np.convolve(loss, np.ones(width) / width, mode="valid"),
            label=label,
            color=line.get_color(),
        )
    for axis, title in zip(axes, ("Single-video smoke", "Fresh dual-seed training")):
        axis.set(xlabel="Optimizer updates", ylabel="Paired reconstruction loss", title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    return paths


def quality_plot(directory, output):
    seeds = available_results(directory)
    if not seeds:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    measures = (
        ("psnr_global_mse", "Global-MSE PSNR / dB"),
        ("lpips", "Full-frame LPIPS-Alex"),
        ("dists", "Full-frame DISTS"),
        ("chroma_retention", "BT.709 chroma variance ratio"),
    )
    for seed in seeds:
        for axis, (name, ylabel) in zip(axes.flat, measures):
            axis.plot(
                BUDGETS,
                [seed["summary"]["paired"][str(k)][name] for k in BUDGETS],
                marker="o",
                label=f"seed {seed['seed']}",
            )
            axis.set(
                xlabel="Displayed token prefix K (no bitstream rate)",
                ylabel=ylabel,
                xscale="log",
                xticks=BUDGETS,
            )
            axis.set_xticklabels([str(k) for k in BUDGETS])
            axis.grid(alpha=0.2)
    for axis in axes.flat:
        axis.legend()
    fig.savefig(output / "gate1_quality_curves.png", dpi=180)
    plt.close(fig)
    return [directory / f"gate1_seed{s['seed']}/evaluation.json" for s in seeds]


def montage(directory, output, seed, video, crop):
    images = directory / f"gate1_seed{seed}/images"
    filename = f"{video}_seed{seed}_{'crop' if crop else 'full'}_comparison.png"
    return comparison(images, output / filename, video, crop)


def comparison(images, output, video, crop):
    suffix = "_crop" if crop else ""
    labels = ["Original", *[f"K={k}" for k in BUDGETS]]
    paths = [
        images / f"{video}_original{suffix}.png",
        *[images / f"{video}_k{k}{suffix}.png" for k in BUDGETS],
    ]
    if not all(p.exists() for p in paths):
        return []
    loaded = [Image.open(p).convert("RGB") for p in paths]
    width, height = loaded[0].size
    header = 24
    canvas = Image.new("RGB", (width * 3, (height + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, frame) in enumerate(zip(labels, loaded)):
        x, y = index % 3 * width, index // 3 * (height + header)
        draw.text((x + 8, y + 5), label, fill="black")
        canvas.paste(frame, (x, y + header))
    canvas.save(output)
    return paths


def fix_horizon_plot(cases, output):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5.5), constrained_layout=True)
    for label, path in cases.items():
        rows = json.loads(path.read_text())["rows"]
        frames = [r["frame"] for r in rows]
        line = axes[0].plot(frames, [r["reference_latent_rms"] for r in rows], label=label)[0]
        axes[0].scatter([31, 63], [rows[i]["reference_latent_rms"] for i in (31, 63)], color=line.get_color())
        axes[0].plot(
            frames,
            [r["native_aligned_latent_rms"] for r in rows],
            color=line.get_color(),
            linestyle=":",
            alpha=0.6,
            label=f"{label} native",
        )
        axes[1].plot(frames, [-10 * np.log10(max(r["mse"], 1e-12)) for r in rows], label=label)
    axes[0].axhline(2, color="gray", linestyle="--", label="RMS limit")
    for ax in axes:
        ax.axvline(32, color="gray", linewidth=1, alpha=0.6)
        ax.axvspan(48, 63, color="gray", alpha=0.08)
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Aligned latent RMS", title="car-roundabout · 64 frames · GOP32")
    axes[1].set(ylabel="K8192 frame PSNR (dB)", xlabel="Frame index", xticks=(0, 15, 31, 47, 63))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def horizon_series(rows):
    return [
        [-10 * np.log10(r["mse"]) for r in rows],
        [r["reference_latent_rms"] for r in rows],
        [r["anchor_min_dimension_entropy"] for r in rows],
        [r["anchor_boundary_fraction"] for r in rows],
    ]


def horizon_plot(directory, output):
    path = directory / "gate1_horizon_diagnostics.json"
    if not path.exists():
        return []
    results = json.loads(path.read_text())["results"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for mode in results:
        rows = mode["rows"]
        frames = [r["frame"] for r in rows]
        series = horizon_series(rows)
        for axis, values in zip(axes.flat, series):
            axis.plot(frames, values, label=mode["mode"])
    labels = (
        "Frame PSNR / dB",
        "Reference latent RMS",
        "Minimum scalar entropy / bit",
        "Anchor boundary fraction",
    )
    for axis, label in zip(axes.flat, labels):
        axis.set(xlabel="Frame index", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    fig.savefig(output / "horizon_diagnostics.png", dpi=180)
    plt.close(fig)
    return [path]


def build():
    directory = resolve("results/aip_vtok")
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    sources = (
        training_plot(directory, output) + quality_plot(directory, output) + horizon_plot(directory, output)
    )
    for seed in (1, 2):
        for crop in (False, True):
            sources += montage(directory, output, seed, "car-roundabout", crop)
    write_json(output / "sources.json", {str(p.relative_to(resolve("."))): file_hash(p) for p in sources})


if __name__ == "__main__":
    build()
