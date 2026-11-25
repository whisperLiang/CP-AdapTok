"""Draw the UF-V2 report figures directly from archived JSON, CSV, and native samples."""

import argparse
import json
from itertools import accumulate
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import ScalarFormatter

from scripts.hla_tok.uf_v2.acceptance import FINAL_EVIDENCE
from scripts.hla_tok.v2.protocol import csv_at

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 145,
})
PALETTE = ("#274e69", "#2d8071", "#c56039", "#946d9b", "#69747c", "#924b43", "#69689b")
BUDGETS = (5, 10, 20, 40)
FIG_WIDTH_MM = 183
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4


class Figures:
    def __init__(self, output, qa):
        self.output = output
        self.qa = qa
        output.mkdir(parents=True, exist_ok=True)
        if qa is not None:
            qa.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name):
        if self.qa is not None:
            from audit_panel_alignment import require_matplotlib_panel_alignment
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=self.qa / f"{name}.layout.json",
                                               tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(self.output / f"{name}.pdf", dpi=600)
        fig.savefig(self.output / f"{name}.svg", dpi=600)
        fig.savefig(self.output / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    def lines(self, name, x, series, xlabel, ylabel, legend="best", ranges=None):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.35), layout="constrained")
        for n, (label, ys) in enumerate(series.items()):
            ax.plot(x, ys, "o-", color=PALETTE[n % len(PALETTE)], lw=1.25, ms=3.8, label=label)
            if ranges and label in ranges:
                ax.errorbar(x, ys, yerr=ranges[label], fmt="none", color=PALETTE[n % len(PALETTE)], capsize=2)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(x)
        ax.grid(axis="y", lw=0.35, color="#cbd2d9", alpha=.6)
        if legend == "outside":
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), fontsize=6.7, ncols=3)
        elif legend:
            ax.legend(loc=legend, fontsize=7)
        self.save(fig, name)

    def bars(self, name, labels, series, xlabel, threshold=None):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, max(2.8, 1.4 + .32 * len(labels))), layout="constrained")
        center = np.arange(len(labels))
        width = .75 / max(1, len(series))
        for n, (method, values) in enumerate(series.items()):
            ax.barh(center + (n - (len(series) - 1) / 2) * width, values, height=width * .92,
                    label=method, color=PALETTE[n % len(PALETTE)])
        ax.set(yticks=center, yticklabels=labels, xlabel=xlabel)
        ax.invert_yaxis()
        if threshold:
            ax.axvline(threshold, color=PALETTE[5], ls=":", lw=1.1)
        if len(series) > 1:
            fig.get_layout_engine().set(h_pad=.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1, 1.16), fontsize=6.75, ncols=len(series))
        self.save(fig, name)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def layout_plot(figs, name, rows):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.72 + .76 * len(rows)), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, len(rows)))
    ax.set_axis_off()
    for index, (label, boxes) in enumerate(rows):
        y = len(rows) - index - .52
        ax.text(.01, y, label, ha="left", va="center", fontweight="bold", size=7.5)
        xs = np.linspace(.19, .82, len(boxes))
        for i, (x, label) in enumerate(zip(xs, boxes, strict=True)):
            patch = FancyBboxPatch((x, y - .25), .16, .50, boxstyle="round,pad=0.005,rounding_size=0.027",
                                   facecolor="#e8f1f2" if index % 2 else "#eef2f5", edgecolor=PALETTE[index % 3],
                                   linewidth=.65)
            ax.add_patch(patch)
            ax.text(x + .08, y, label, ha="center", va="center", size=6.7)
            if i:
                ax.annotate("", (x - .005, y), (xs[i - 1] + .164, y),
                            arrowprops={"arrowstyle": "->", "color": "#56616a", "lw": .62})
    figs.save(fig, name)


def architectures(figs):
    layout_plot(figs, "fig01_architecture", [
        ("Sender", ("Native UF-LD\nanalysis", "Base and\nlatent delta", "Rank and\npack records", "Slice physical\nbyte prefix")),
        ("Receiver", ("Commit native\nBase", "Append integer\nresiduals", "Refine cached\nI/P feature", "Native RGB\nreadout")),
        ("Next frame", ("Base I/P\nreference", "Native temporal\nmemory", "Prepare next\ncontext", "Next frame\nanalysis")),
    ])
    layout_plot(figs, "fig02_decoders", [
        ("Anchor-Fast", ("Native Base\nRGB", "RGB residual\nlatent", "RGB residual\nhead", "Sum and clamp")),
        ("D0", ("Native Base\nlatent", "Add latent\nresidual", "Entire native\ndecoder", "RGB readout")),
        ("D1", ("Native Base\nfeature", "Project latent\nresidual", "Two DepthConv\nblocks", "Shared RGB\nreadout")),
    ])


def profiler(figs, results, report, final):
    profile = read(results / "profiling/legacy.json")["profile"]["d0"]["phase"]
    encode = profile["enh.build_inclusive"]["ms_per_frame"]
    group = profile["enh.group_and_score"]["ms_per_frame"]
    quant = profile["enh.quantize_and_copy"]["ms_per_frame"]
    rans = profile["enh.rans_encode"]["ms_per_frame"]
    names = ("Quantization + D2H", "Group and score", "rANS encode", "Remaining grouping/packing",
             "Parse and accumulate", "Display call")
    vals = (quant, group, rans, encode - group - quant - rans,
            profile["enh.apply"]["ms_per_frame"], profile["display.render"]["ms_per_frame"])
    figs.bars("fig03_d0_hotspots", names, {"Instrumented": vals}, "Wall time per frame (ms)")
    historical = read(results / "baseline/baseline_audit.json")["curve_sources"]["uf_anchor"]["rows"]
    old = [r["total_ms_p95"] for r in historical if r["budget_mbps"] in BUDGETS]
    new = [report["measured"]["Anchor-Fast"]["timing"][str(b)]["pooled"]["total_ms_p95"] for b in BUDGETS]
    figs.lines("fig04_anchor_latency", BUDGETS, {"Old Anchor": old, "Anchor-Fast": new}, "Frame cap (Mbps)",
               "End-to-end P95 (ms)")
    d0 = [r["serial_fps"] for r in report["curves"]["UF-Progressive D0"] if r["budget_mbps"] in BUDGETS]
    speed = final["timing"]
    smoke = read(results / "d1/zero-parallel/summary.json")["timing"]
    d1 = "Selected D1, range over 3 repeats"
    fps = [speed[str(b)]["pooled"]["fps"] for b in BUDGETS]
    repeats = [[r["fps"] for r in speed[str(b)]["repetitions"]] for b in BUDGETS]
    bounds = [[center - min(values) for center, values in zip(fps, repeats, strict=True)],
              [max(values) - center for center, values in zip(fps, repeats, strict=True)]]
    figs.lines("fig05_d0_d1_speed", BUDGETS, {
        "Legacy D0 (174 frames)": d0,
        "D1 step 0 (34 frames)": [smoke[str(b)]["pooled"]["fps"] for b in BUDGETS],
        d1: fps,
    }, "Frame cap (Mbps)", "End-to-end FPS", legend="outside", ranges={d1: bounds})


def rate_distortion(figs, report, selected):
    metrics = {"global_psnr": ("fig06_psnr_rd", "Global RGB PSNR (dB)"),
               "lpips": ("fig07_lpips_rd", "LPIPS-Alex"),
               "dists": ("fig08_dists_rd", "DISTS")}
    curated = ("Original AdapTok", "Official UF-LD", "HLA-Tok-T", "CC-N", "Anchor-Fast", selected)
    for metric, (name, ylabel) in metrics.items():
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.55), layout="constrained")
        for index, method in enumerate(curated):
            points = sorted(report["curves"][method], key=lambda row: row["actual_mbps"])
            ax.plot([r["actual_mbps"] for r in points], [r[metric] for r in points],
                    c=PALETTE[index], label=method, ms=3.5, marker="o",
                    ls="--" if index < 2 else "-", lw=1.6 if method == selected else .88)
        ax.set(xlabel="Actual application rate (Mbps)", ylabel=ylabel, xscale="log")
        ax.set_xticks([.25, .5, 1, 2, 5, 10, 20, 40])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(loc="best", fontsize=6.75, ncols=2)
        figs.save(fig, name)


def equal_rate(figs, report, selected):
    to_display = ("Original AdapTok", "Official UF-LD", "CC-N", "Anchor-Fast", selected)
    by_key = {(r["method"], r["metric"], r["target"]): r["candidate_actual_mbps"]
              for r in report["equal_quality"]}
    for index, (metric, targets, label) in enumerate([
        ("global_psnr", (25, 28, 30), "PSNR (dB)"), ("lpips", (.28978,), "LPIPS-Alex")
    ]):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.35), layout="constrained")
        for column, method in enumerate(to_display):
            rates = [by_key.get((method, metric, target)) for target in targets]
            offset = (column - 2) * .11
            x, y = zip(*[(j + offset, rate) for j, rate in enumerate(rates) if rate is not None], strict=True) if (
                any(r is not None for r in rates)) else ([], [])
            ax.scatter(x, y, color=PALETTE[column], marker="o", s=23, label=method)
        ax.set(xticks=range(len(targets)), xticklabels=targets, xlabel=label, ylabel="Actual Mbps at target",
               yscale="log", xlim=(-.45, len(targets) - .55))
        ax.legend(loc="best", fontsize=6.75)
        if index:
            ax.add_artist(AnchoredText("Unreached targets omitted", loc="lower right", prop={"size": 6.4},
                                       frameon=False))
        figs.save(fig, ("fig09_equal_psnr", "fig10_equal_lpips")[index])


def base_and_bytes(figs, root, final):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.45), layout="constrained")
    labels = []
    for label in ("base-standard", "base-lite"):
        rows = csv_at(root / "diagnostics/base-validation" / f"{label}.csv")
        for kind in ("I", "P"):
            labels.append((f"{label.title()} {kind}",
                           np.sort([int(r["application_bytes"]) for r in rows if r["type"] == kind])))
    for index, (label, values) in enumerate(labels):
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label, c=PALETTE[index])
    ax.axvline(4096, c=PALETTE[5], ls=":", lw=1.1)
    ax.set(xlabel="Base application bytes", ylabel="Empirical cumulative fraction")
    ax.legend(loc="lower right", fontsize=6.75)
    figs.save(fig, "fig11_base_bytes")

    entropy = read(root / "entropy/summary.json")["same_symbol"]
    figs.lines("fig12_cc_s_savings", BUDGETS,
               {"Static CDF E1": [100 * v["e1_savings_fraction"] for v in entropy.values()],
                "Static plus Raw E2": [100 * v["e2_savings_fraction"] for v in entropy.values()]},
               "Matched E0 prefix cap (Mbps)", "Application bytes saved (%)")
    path = root / final["summary"]
    rows = csv_at(path.with_name("quality.csv"))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.25), layout="constrained")
    bottoms = np.zeros(len(BUDGETS))
    for i, (column, label) in enumerate((
        ("native_joint_y_z_bytes", "Native joint y/z"), ("base_header_bytes", "Base header and CRC"),
        ("enhancement_headers", "Record headers and CRC"), ("enhancement_payload", "Enhancement payload")
    )):
        sizes = [np.mean([float(r[column]) for r in rows if float(r["budget_mbps"]) == b]) for b in BUDGETS]
        ax.bar(np.arange(4), sizes, bottom=bottoms, label=label, color=PALETTE[i])
        bottoms += sizes
    ax.set(xticks=np.arange(4), xticklabels=BUDGETS, xlabel="Frame cap (Mbps)", ylabel="Bytes per frame")
    ax.legend(fontsize=6.75, loc="upper left")
    figs.save(fig, "fig13_application_byte_breakdown")


def images_and_diagnostics(figs, root, selected):
    locations = root / "diagnostics/perceptual"
    diagnostics = read(locations / "summary.json")["scenarios"]
    keys = ("Q1_base", "Q5_d0_5", "Q5_d0_full", "Q4_continuous_bypass", "Q3_5_anchor_seed1",
            "Q8_true_luma", "Q8_true_chroma", "Q9_true_high_frequency", "Q9_true_low_frequency")
    names = ("Base", "D0 at 5 Mbps", "D0 maximum prefix", "Continuous latent", "Anchor at 5 Mbps",
             "Replace luma", "Replace chroma", "Replace high frequencies", "Replace low frequencies")
    figs.bars("fig14_perceptual_diagnostics", names,
              {"LPIPS": [diagnostics[k]["lpips"] for k in keys]}, "LPIPS-Alex on 12 native frames")

    samples = root / "samples" / FINAL_EVIDENCE[selected][2]
    manifest = read(samples / "manifest.json")
    if manifest is None:
        return
    sample_names = ("Input", "Anchor-Fast", "D0", "D1")
    x0, y0, x1, y1 = manifest["crop_xyxy"]
    for crop in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 4.42), layout="constrained")
        for ax, row, name in zip(axes.flat, manifest["samples"], sample_names, strict=True):
            image = plt.imread(samples / row["file"])
            ax.imshow(image[y0:y1, x0:x1] if crop else image, interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(name, loc="left", fontsize=7.5)
        figs.save(fig, "fig16_native_crop" if crop else "fig15_native_comparison")


def training(figs, root):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.25), layout="constrained")
    for seed, linestyle in ((1, "-"), (2, "--")):
        p1 = root / f"training/seed{seed}"
        logs = [p1 / "steps-1-500.csv", p1 / "steps-501-2000.csv"]
        joint = list((root / f"training/joint-seed{seed}").glob("steps-*.csv"))
        joint.extend((root / f"training/formal-seed{seed}").glob("steps-*.csv"))
        for col, (files, label) in enumerate(((logs, "Distillation"), (joint, "Joint RD"))):
            rows = {int(row["step"]): row for file in files for row in csv_at(file)}
            ordered = [rows[n] for n in sorted(rows)]
            bins = [ordered[i:i + 50] for i in range(0, len(ordered), 50)]
            if bins:
                ax.plot([int(group[-1]["step"]) for group in bins],
                        [np.mean([float(r["lpips"]) for r in group]) for group in bins],
                        c=PALETTE[col], ls=linestyle, label=f"{label}, seed {seed}", lw=1)
    ax.set(xlabel="Cumulative optimizer steps", ylabel="Train crop LPIPS-Alex")
    ax.legend(fontsize=7)
    figs.save(fig, "fig17_training")


def ablation(figs, root):
    entries = {
        "Fixed scan E0": "d1/seed1-step2000-fixed",
        "Fixed scan E1": "d1/seed1-step2000-fixed-static",
        "Residual ordering E0": "d1/seed1-step2000-validation",
        "Residual ordering E1": "d1/seed1-step2000-static",
        "Perceptual weighting E1": "d1/seed1-step2000-perceptual",
    }
    series = {}
    for label, directory in entries.items():
        evidence = read(root / directory / "summary.json")
        if evidence:
            rows = {r["budget_mbps"]: r for r in evidence["quality"]}
            series[label] = [rows[b]["lpips"] for b in BUDGETS]
    figs.lines("fig18_perceptual_ordering", BUDGETS, series, "Frame cap (Mbps)", "LPIPS-Alex")


def dynamic_and_pareto(figs, root, report, final, selected):
    dynamic = csv_at(root / "evaluation" / FINAL_EVIDENCE[selected][0] / "frames.csv")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.2), layout="constrained")
    for n, video in enumerate(dict.fromkeys(r["video"] for r in dynamic)):
        rows = [r for r in dynamic if r["video"] == video]
        mismatch = list(accumulate(r["base_state_equal"] != "True" for r in rows))
        ax.step(np.arange(len(rows)), mismatch, where="post", ls=("-", "--")[n], lw=1.2,
                label=video.split(":")[-1] + f" ({len(rows)} frames)", color=PALETTE[n])
    ax.set(xlabel="Frame index with changing bandwidth and enhancement loss",
           ylabel="Cumulative reference mismatches", ylim=(-.1, 1.1), yticks=(0, 1))
    ax.legend(loc="upper left", fontsize=7)
    figs.save(fig, "fig19_causal_reference")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.75), layout="constrained")
    for n, method in enumerate(("Anchor-Fast", selected)):
        method_data = report["measured"][method] if method != selected else final
        samples = [r for r in method_data["quality"] if r["budget_mbps"] in BUDGETS]
        fps = [method_data["timing"][str(int(r["budget_mbps"]))]["pooled"]["fps"] for r in samples]
        ax.scatter([r["actual_mbps"] for r in samples], [r["lpips"] for r in samples],
                   s=[f * .8 for f in fps], color=PALETTE[n], label=method, alpha=.85,
                   marker=("o", "s")[n])
    ax.set(xlabel="Actual application rate (Mbps)", ylabel="LPIPS-Alex", xscale="log")
    ax.set_xticks([5, 10, 20, 40])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(fontsize=7)
    ax.add_artist(AnchoredText("Marker area scales with throughput", loc="lower left", prop={"size": 6.4},
                               frameon=False))
    figs.save(fig, "fig20_quality_rate_throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--selected", required=True, choices=FINAL_EVIDENCE)
    parser.add_argument("--qa", type=Path)
    args = parser.parse_args()
    root = args.results
    report = read(root / "evaluation/rate_analysis.json")
    final = report["measured"][args.selected]
    figures = Figures(root / "figures", args.qa)
    architectures(figures)
    profiler(figures, root, report, final)
    rate_distortion(figures, report, args.selected)
    equal_rate(figures, report, args.selected)
    base_and_bytes(figures, root, final)
    images_and_diagnostics(figures, root, args.selected)
    training(figures, root)
    ablation(figures, root)
    dynamic_and_pareto(figures, root, report, final, args.selected)


if __name__ == "__main__":
    main()
