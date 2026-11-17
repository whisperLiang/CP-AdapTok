"""Plot only observed experiment records; no smoothing or synthetic observations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cp_adaptok.runtime import write_json

PALETTE = ["#355C7D", "#B76D45", "#7A8260", "#856788"]


def read(path):
    return json.loads(Path(path).read_text())


def export(fig, path):
    fig.canvas.draw()
    if len(fig.axes) != 1:
        raise ValueError("these plots use one axes; multi-panel plots require an alignment audit")
    write_json(
        str(path) + ".alignment.json", {"status": "not_applicable", "reason": "one rendered plot area"}
    )
    fig.savefig(f"{path}.png", dpi=600)
    fig.savefig(f"{path}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.svg")
    svg = Path(f"{path}.svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def axes():
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.8)
    return fig, ax


def training_curves(root):
    fig, ax = axes()
    data = []
    for color, length in zip(PALETTE, (1, 4, 16)):
        history = read(root / "overfit" / f"{length}frame/history.json")
        steps = [0] + [r["step"] for r in history["train"]]
        values = [history["step0_train"]["summary"]["global_mse_psnr"]] + [
            r["psnr"] for r in history["train"]
        ]
        ax.plot(steps, values, color=color, label=f"{length}-frame fit", linewidth=1.2)
        data.extend({"frames": length, "step": step, "psnr": value} for step, value in zip(steps, values))
    ax.set(
        xlabel="Completed optimizer updates",
        ylabel="Training PSNR (dB)",
        xlim=(0, 300),
        xticks=[0, 100, 200, 300],
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3)
    write_json(root / "figures/training_curves.source.json", data)
    export(fig, root / "figures/training_curves")


def paired_quantization(root):
    fig, ax = axes()
    entries = [
        ("B1 S4 / Original128", "baseline", "paired"),
        ("R1 trained / Full2048", "continuous/evaluation", "paired_full2048"),
        ("R2 trained / Full2048", "quantized/evaluation", "paired_full2048"),
    ]
    data = []
    for index, (label, directory, reference) in enumerate(entries):
        values = [
            r for r in read(root / directory / "paired_quantization.json") if r["reference"] == reference
        ]
        if not values:
            raise ValueError("missing paired frame measurements")
        delta = [r["r2_minus_r1_psnr"] for r in values]
        jitter = np.linspace(-0.12, 0.12, len(delta))
        ax.scatter(index + jitter, delta, color=PALETTE[index], s=14, alpha=0.7, label=label)
        data.extend(dict(r, checkpoint_state=label) for r in values)
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["B1 S4", "R1 trained", "R2 trained"],
        ylabel="Paired VQ minus continuous PSNR (dB)",
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=7, ncol=1)
    write_json(root / "figures/paired_quantization.source.json", data)
    export(fig, root / "figures/paired_quantization")


def select_reference_rows(rows, path, reference):
    return sorted(
        [r for r in rows if r["path"] == path and r["reference"] == reference],
        key=lambda row: (row["video"], row["frame"]),
    )


def reference_frames(root):
    fig, ax = axes()
    source = root / "continuous/evaluation/frame_metrics.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    videos = sorted({r["video"] for r in rows})
    data = []
    for index, (path, reference) in enumerate(
        (("R1", "full2048"), ("R1", "original128"), ("R2", "full2048"), ("R2", "original128"))
    ):
        selected = select_reference_rows(rows, path, reference)
        ax.plot(
            range(len(selected)),
            [r["psnr"] for r in selected],
            color=PALETTE[index],
            linestyle="--" if reference == "original128" else "-",
            label=f"{path} / {reference}",
            linewidth=1,
        )
        data.extend(selected)
    for boundary in (7.5, 15.5):
        ax.axvline(boundary, color="#BBBBBB", linewidth=0.6)
    ax.set(
        xlabel="Validation frame (three independent 8-frame videos)",
        ylabel="PSNR (dB)",
        xticks=[3.5, 11.5, 19.5],
        xticklabels=[v.split(":")[-1] for v in videos],
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=7)
    write_json(root / "figures/reference_frames.source.json", data)
    export(fig, root / "figures/reference_frames")


def build_figures(root):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    (root / "figures").mkdir(parents=True, exist_ok=True)
    training_curves(root)
    paired_quantization(root)
    reference_frames(root)
