"""Export the real V3 results and retain figure-level data provenance."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from audit_figure_collisions import audit_pdf as audit_collisions
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_pdf_text import audit_pdf as audit_fonts

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

BLUE, INK, ORANGE, GRAY, TEAL = "#28619a", "#203548", "#cb7243", "#868e96", "#29867b"
CORE = (BLUE, ORANGE, GRAY, TEAL)
BUDGETS = (5, 10, 20, 40)
METRICS = (("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS-Alex"), ("dists", "DISTS"))
V2 = ROOT / "results/hla_tok/uf_v2"

mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.linewidth": 0.65, "legend.frameon": False,
                    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.labelcolor": INK,
                    "xtick.color": INK, "ytick.color": INK, "text.color": INK})


class Export:
    def __init__(self, root, qa):
        self.root, self.qa = root, qa
        self.entries = []
        self.audits = {}
        root.mkdir(parents=True, exist_ok=True)
        qa.mkdir(parents=True, exist_ok=True)

    def write(self, fig, name, inputs):
        alignment = require_matplotlib_panel_alignment(fig, json_out=self.qa / f"{name}.alignment.json",
                                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        pdf = self.root / f"{name}.pdf"
        svg = self.root / f"{name}.svg"
        tiff = self.root / f"{name}.tiff"
        fig.savefig(pdf, format="pdf")
        font = audit_fonts(pdf.read_bytes(), minimum_pt=5)
        collisions = audit_collisions(pdf)
        write_json(self.qa / f"{name}.text.json", font)
        write_json(self.qa / f"{name}.collisions.json", collisions)
        if not font["auditable"] or font["below_minimum_count"] or collisions["verdict"] != "PASS":
            raise ValueError(f"Exported figure failed text geometry checks: {name}")
        self.audits[name] = {"alignment": alignment["verdict"], "collisions": collisions["verdict"],
                             "minimum_found_pt": font["minimum_found_pt"]}
        fig.savefig(svg, format="svg")
        svg.write_text("\n".join(line.rstrip(" \t") for line in svg.read_text().split("\n")))
        fig.savefig(tiff, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        names = (pdf, svg, tiff)
        self.entries.append({"plot": name, "files": {p.name: sha256_file(p) for p in names},
                             "data": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in inputs]})
        plt.close(fig)


def canvas(columns=1, height=2.5):
    if columns == 1:
        fig, axes = plt.subplots(figsize=(3.50, height), layout="constrained")
    else:
        fig, axes = plt.subplots(1, columns, figsize=(7.20, height))
        margins = {"left": .18, "right": .98, "bottom": .23, "top": .87, "wspace": .72}
        if columns == 3:
            margins.update(left=.095, wspace=.48)
        if columns > 3:
            margins.update(left=.01, right=.99, bottom=.10, top=.81, wspace=.08)
        fig.subplots_adjust(**margins)
    return fig, np.atleast_1d(axes)

def architecture(export, decision):
    fig, ax = plt.subplots(figsize=(7.20, 1.8), layout="constrained")
    ax.set_axis_off()
    contents = (("V2 fit", "Native Base cache", "Crop features & residuals", "D1 + P gains"),
                ("V3 fit", "Current RGB + Base history", "Native Encoder, prior & Decoder", "Joint RGB and rate"))
    for y, row, color in zip((0.71, .27), contents, (GRAY, BLUE), strict=True):
        for x, label in zip((.09, .33, .60, .86), row, strict=True):
            ax.text(x, y, label, ha="center", va="center", transform=ax.transAxes,
                    color=INK, bbox={"boxstyle": "round,pad=0.48", "facecolor": "white", "edgecolor": color})
        for start, end in ((.145, .205), (.457, .465), (.736, .768)):
            ax.annotate("", xy=(end, y), xytext=(start, y), xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": color, "lw": .7})
    export.write(fig, "fig01_training_paths", [decision])


def capacity(export, path, frame_csv):
    data = json_at(path)["results"]["2"]["pooled"]
    rows = [data[k] for k in ("C0/CONTINUOUS", "C1/CONTINUOUS", "C2/ALL_VALID_RECORDS", "C3/ALL_VALID_RECORDS")]
    fig, axes = canvas(3)
    for (field, label), ax in zip(METRICS, axes, strict=True):
        ax.scatter(["C0", "C1", "C2", "C3"], [row[field] for row in rows], color=CORE, s=40)
        ax.set_ylabel(label)
    export.write(fig, "fig02_fixed_history_capacity", [path])
    frames = [r for r in csv_at(frame_csv) if r["scenario"] == "C3" and r["budget"] == "ALL_VALID_RECORDS"]
    fig, (ax,) = canvas()
    observed = np.array([int(r["bytes"]) * .00024 for r in frames])
    ax.hist(observed, bins=16, color=BLUE)
    ax.axvline(40, color=ORANGE, lw=1.3, linestyle="--")
    ax.set(xlabel="All valid records (Mbps at 30 fps)", ylabel="Frames")
    export.write(fig, "fig06_maximum_effective_capacity", [frame_csv])


def gradients(export, audit, runs):
    fig, axes = canvas(2)
    history = json_at(audit)["replay"]["P3"]["gradient_l2"]
    names = ("encoder_logits", "reconstruction_logits", "hyper_logits", "quantizer.inter.logit")
    labels = ("P Encoder gain", "P recon. gain", "P hyper bias", "P quantizer")
    x = np.arange(len(names))
    for name, color, offset in (("distortion", BLUE, -.18), ("rate", ORANGE, .18)):
        axes[0].barh(x + offset, [history[name][key] for key in names], height=.34, color=color, label=name.capitalize())
    axes[0].set(yticks=x, yticklabels=labels, xlabel="Gradient L2", title="V2 P3 replay", xlim=(0, .12))
    axes[0].set_xscale("symlog", linthresh=1e-5)
    axes[0].set_xticks([0, 1e-4, 1e-3, 1e-2, .1], ["0", "1e-4", "1e-3", "1e-2", "1e-1"])
    axes[0].legend(loc="lower right")
    native = [path for path in runs if path.parent.name.startswith("probe-") or path.parent.name.startswith("P3-")]
    if native:
        result = json_at(native[-1])
        picks = {"I Encoder": ("native_i", "enc."), "P Encoder": ("native_p", "encoder."),
                 "P Decoder": ("native_p", "decoder."), "P readout": ("native_p", "recon_head.")}
        data = [np.sqrt(sum(row["l2"] ** 2 for name, row in result["delta"][part].items()
                            if name.startswith(prefix))) for part, prefix in picks.values()]
        axes[1].barh(np.arange(len(data)), data, color=BLUE)
        axes[1].set(yticks=np.arange(len(data)), yticklabels=list(picks), xlabel="Parameter update L2",
                    title="V3 joint updates")
    else:
        axes[1].text(.5, .5, "Joint-run evidence unavailable", ha="center", transform=axes[1].transAxes)
        axes[1].set(xlabel="Parameter update L2", title="V3 joint updates")
    export.write(fig, "fig03_gradient_and_parameter_evidence", [audit, *native[-1:]])


def runtime(export, cap, full, demand):
    fast = json_at(cap)["results"]["d0_fast"]
    v2 = V2 / "d1/formal-seed2-final-e0/summary.json"
    d1 = json_at(v2)["timing"]
    fig, axes = canvas(2)
    for ax, (field, label, threshold) in zip(axes, (("fps", "Frames / s", 30), ("total_ms_p95", "P95 (ms)", 33.33)),
                                            strict=True):
        ax.plot(BUDGETS, [fast[str(b)]["pooled"][field] for b in BUDGETS], "s-", color=GRAY, label="D0-Fast")
        ax.plot(BUDGETS, [d1[str(b)]["pooled"][field] for b in BUDGETS], "o-", color=BLUE, label="V2 D1")
        ax.scatter([5], [6.66 if field == "fps" else 249.39], color=ORANGE, marker="x", label="Historical D0")
        ax.axhline(threshold, color=INK, lw=.7, ls=":")
        ax.set(xlabel="Hard cap (Mbps)", ylabel=label, xticks=BUDGETS)
        if field != "fps":
            ax.set_yscale("log")
    axes[0].legend(loc="lower right")
    export.write(fig, "fig04_native_decoder_timing", [cap, v2])
    if not full.exists():
        return
    data = json_at(full)["timing"]
    fig, (ax,) = canvas()
    components = (("native_encode_ms", "Native encode"), ("analysis_copy_ms", "Quantize and copy"),
                  ("group_rank_pack_ms", "Rank and pack"), ("base_ready_ms", "Base decode"),
                  ("enhancement_ms", "Enhancement decode"), ("render_ms", "Display"))
    offset = np.zeros(len(BUDGETS))
    for i, (field, label) in enumerate(components):
        values = [data[str(b)]["pooled"][f"{field}_mean"] for b in BUDGETS]
        ax.bar([str(b) for b in BUDGETS], values, bottom=offset, label=label,
               color=(BLUE, "#a1bed6", TEAL, "#a8cbbd", ORANGE, "#e5b59f")[i])
        offset += values
    if demand.exists():
        d = json_at(demand)["timing"]
        ax.plot(range(len(BUDGETS)), [d[str(b)]["pooled"]["total_ms_mean"] for b in BUDGETS],
                color=INK, marker="D", label="On-demand total")
    ax.set(xlabel="Hard cap (Mbps)", ylabel="Mean elapsed (ms)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.24), ncol=2, fontsize=7)
    export.write(fig, "fig17_runtime_breakdown", [full, *([demand] if demand.exists() else [])])


def entropy(export, path):
    data = json_at(path)["measurements"]
    fig, (volume, savings) = canvas(2)
    budgets = ["5", "10", "20", "40", "ALL_VALID_RECORDS"]
    positions = np.arange(len(budgets))
    for mode, offset, color in (("E0", -.18, GRAY), ("E1", .18, BLUE)):
        selected = [data[mode][name]["same_symbol"]["bytes"] / 1e6 for name in budgets]
        volume.bar(positions + offset, selected, width=.34, color=color, label=mode)
    for metric, color, marker, label in (("whole_savings", BLUE, "o", "Application bytes"),
                                         ("enhancement_savings", ORANGE, "s", "Enhancement bytes")):
        attempts = [[100 * data[mode][name]["same_symbol"][metric] for name in budgets] for mode in ("E1", "E2")]
        if attempts[0] == attempts[1]:
            savings.plot(positions, attempts[0], marker=marker, color=color, label=label)
        else:
            for mode, values in zip(("E1", "E2"), attempts, strict=True):
                savings.plot(positions, values, marker=marker, color=color, label=f"{mode} {label.lower()}")
    for axes in (volume, savings):
        axes.set(xticks=positions, xticklabels=["5", "10", "20", "40", "All valid"], xlabel="E0 record support")
        axes.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=2, borderaxespad=0, fontsize=7)
    volume.set(ylabel="Application bytes (millions)")
    savings.set(ylabel="Same-symbol savings (%)")
    export.write(fig, "fig05_static_entropy_bytes", [path])


def rd(export, path):
    curves = json_at(path)["curves"]
    names = ("Original AdapTok", "Official UF-LD", "Anchor-Fast", "V2 D1", "V3 joint")
    for index, (metric, label) in enumerate(METRICS, 8):
        fig, (ax,) = canvas()
        for name, color, mark in zip(names, (GRAY, "#746c8f", ORANGE, TEAL, BLUE), ("x", "s", "^", "d", "o"), strict=True):
            points = [r for r in curves.get(name, []) if "actual_mbps" in r]
            if points:
                points.sort(key=lambda row: row["actual_mbps"])
                ax.plot([r["actual_mbps"] for r in points], [r[metric] for r in points],
                        color=color, marker=mark, ms=3, lw=1, label=name)
        ax.set(xscale="log", xlabel="Actual application Mbps", ylabel=label)
        ax.legend(fontsize=7, loc="best")
        export.write(fig, f"fig{index:02d}_{metric}_rd", [path])
    selected = json_at(path)["equal_quality"]
    rows = [r for r in selected if r["metric"] == "lpips" and r["target"] == .28978
            and r["candidate_actual_mbps"] is not None and r["method"] in (*names, "V2 E1")]
    if rows:
        fig, (ax,) = canvas()
        ax.barh([r["method"] for r in rows], [r["candidate_actual_mbps"] for r in rows], color=BLUE)
        ax.set(xlabel="Actual Mbps at LPIPS 0.28978")
        export.write(fig, "fig11_equal_perceptual_rate", [path])
    fig, (ax,) = canvas()
    for name, color in (("V2 D1", TEAL), ("V3 joint", BLUE)):
        rows = sorted((r for r in curves.get(name, []) if r.get("budget_mbps") in BUDGETS and "actual_mbps" in r),
                      key=lambda row: row["budget_mbps"])
        if len(rows) == len(BUDGETS):
            changes = [((a["lpips"] - b["lpips"]) / (b["actual_mbps"] - a["actual_mbps"]))
                       for a, b in zip(rows, rows[1:])]
            ax.plot(["5→10", "10→20", "20→40"], changes, marker="o", color=color, label=name)
    ax.set(xlabel="Adjacent hard cap (Mbps)", ylabel="LPIPS gain / added Mbps")
    ax.legend()
    export.write(fig, "fig12_marginal_gain", [path])


def bases(export, lite, curve):
    paths = [lite / f"validation_{name}.csv" for name in ("L0", "L1")]
    values = {name: [int(r["application_bytes"]) for r in csv_at(path)]
              for name, path in zip(("Lite L0", "Lite L1"), paths, strict=True)}
    rows = csv_at(curve)
    values["Standard"] = [int(r["base_bytes"]) for r in rows if float(r["budget_mbps"]) == 5]
    fig, (ax,) = canvas()
    for (name, items), color in zip(values.items(), (GRAY, ORANGE, BLUE), strict=True):
        ax.plot(np.sort(items), np.arange(1, len(items) + 1) / len(items), label=name, color=color)
    ax.axvline(4096, color=ORANGE, lw=.8, ls=":")
    ax.axvline(9212, color=BLUE, lw=.8, ls=":")
    ax.set(xlabel="Application Base bytes", ylabel="Empirical cumulative fraction")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, borderaxespad=0, fontsize=7)
    export.write(fig, "fig13_base_bytes", [*paths, curve])


def images(export, directory):
    path = directory / "manifest.json"
    if not path.exists():
        return
    files = json_at(path)["images"]
    labels = [row["scenario"] for row in files]
    for index, suffix in ((14, "native"), (15, "crop")):
        fig, axes = canvas(len(files), height=1.8 if index == 14 else 1.7)
        for ax, row, name in zip(axes, files, labels, strict=True):
            image = plt.imread(directory / row["file"])
            image = image if index == 14 else image[400:624, 820:1044]
            ax.imshow(image, interpolation="nearest")
            ax.set_title(name.replace("budget_", "") + (" Mbps" if name.startswith("budget_") else ""))
            ax.set_axis_off()
        export.write(fig, f"fig{index}_{suffix}_reconstruction", [path])


def training(export, paths):
    if not paths:
        return
    fig, (ax,) = canvas()
    for i, path in enumerate(paths):
        rows = csv_at(path)
        ax.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows],
                color=(TEAL, BLUE, ORANGE, GRAY)[i % 4], alpha=.7, lw=.7, label=path.parent.name)
    ax.set(xlabel="Actual optimizer updates", ylabel="Rate–distortion objective")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, borderaxespad=0, fontsize=7)
    export.write(fig, "fig16_training_curve", paths)


def pareto(export, path, full, v2_static):
    fig, (quality, speed) = canvas(2)
    curves = json_at(path)["curves"]
    sources = {"V3 joint": full, "V2 E1": v2_static, "V2 D1": V2 / "d1/formal-seed2-final-e0/summary.json",
               "Anchor-Fast": V2 / "anchor_fast/validation/summary.json"}
    for (name, file), color, marker in zip(sources.items(), (BLUE, TEAL, GRAY, ORANGE), ("o", "s", "d", "^"), strict=True):
        if not file.exists() or name not in curves:
            continue
        times = json_at(file)["timing"]
        rows = [row for row in curves[name] if row.get("budget_mbps") in BUDGETS]
        rates = [row["actual_mbps"] for row in rows]
        fps = [times[str(int(row["budget_mbps"]))]["pooled"]["fps"] for row in rows]
        quality.plot(rates, [row["lpips"] for row in rows], marker=marker, ms=4, lw=.85, color=color, label=name)
        speed.plot(rates, fps, marker=marker, ms=4, lw=.85, color=color)
    quality.set(ylabel="LPIPS-Alex")
    speed.set(ylabel="Full-max frames / s")
    for axes in (quality, speed):
        axes.set(xlabel="Actual application Mbps", xscale="log")
    fig.legend(*quality.get_legend_handles_labels(), loc="upper center", ncol=4)
    export.write(fig, "fig18_quality_rate_fps", [path, *(p for p in sources.values() if p.exists())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()
    out = args.results.resolve()
    export = Export(out / "figures", args.qa)
    cap = out / "diagnostics/validation/capacity.json"
    full, demand = (out / f"evaluation/{mode}/summary.json" for mode in ("A4", "on_demand"))
    verdict_path = out / "final_verdict.json"
    if verdict_path.exists():
        verdict = json_at(verdict_path)
        full, demand = (out / verdict[key] for key in ("full_max_summary", "on_demand_summary"))
    runs = sorted((out / "training").glob("*/summary.json"))
    architecture(export, out / "diagnostics/capacity.json")
    capacity(export, cap, cap.parent / "seed2/frames.csv")
    gradients(export, out / "audit/training_audit.json", runs)
    runtime(export, cap, full, demand)
    entropy(export, out / "entropy/summary.json")
    rd(export, out / "evaluation/rates.json")
    if (out / "base_lite/summary.json").exists() and (out / "evaluation/A4/quality.csv").exists():
        bases(export, out / "base_lite", out / "evaluation/A4/quality.csv")
    images(export, out / "samples/native")
    training(export, sorted((out / "training").glob("*/steps.csv")))
    pareto(export, out / "evaluation/rates.json", full, out / "evaluation/A2/summary.json")
    write_json(export.root / "manifest.json", {"figures": export.entries, "fig07_detail": "NOT_RUN",
        "scope": "measured cohort point estimates; timing points are technical repetitions"})
    write_json(export.root / "qa.json", {"figures": export.audits, "rendered": len(export.audits),
        "fig07_detail": "NOT_RUN", "pdf_text_minimum_required_pt": 5, "status": "PASS"})


if __name__ == "__main__":
    main()
