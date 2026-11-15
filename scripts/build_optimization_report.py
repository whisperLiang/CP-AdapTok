"""Build tables and figures only from recorded measurements; missing gates remain explicit."""

import csv
import json

import numpy as np

from cp_adaptok.runtime import ROOT, write_json

RESULTS = ROOT / "results/optimization_20261115"


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def csv_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_prefix_metric(records, name, index=-1):
    return float(np.mean([row["prefixes"][index][name] for row in records]))


def deterministic_statistics(records):
    counts = np.sum([r["vq"]["deterministic"]["counts"] for r in records], axis=0)
    return {
        "deterministic_perplexity": float(np.mean([r["vq"]["deterministic"]["perplexity"] for r in records])),
        "deterministic_distinct_codes": int((counts > 0).sum()),
    }


def quality_drop_ratio(records):
    regressions = [
        b["mse"] > a["mse"] for row in records for a, b in zip(row["prefixes"], row["prefixes"][1:])
    ]
    return float(np.mean(regressions))


def ablation_summary(path):
    report = read(path)
    records = report["records"]
    deltas = [r["prefixes"][-1]["psnr"] - r["prefixes"][0]["psnr"] for r in records]
    return {
        "experiment": path.parent.name,
        "updates": 100,
        "frames_per_update": len(records),
        "continuous_diagnostic": report["continuous_only_diagnostic"],
        "base_psnr": mean_prefix_metric(records, "psnr", 0),
        "max_psnr": mean_prefix_metric(records, "psnr"),
        "delta_psnr": float(np.mean(deltas)),
        **deterministic_statistics(records),
        "quality_drop_ratio": quality_drop_ratio(records),
        "max_lpips": mean_prefix_metric(records, "lpips"),
        "max_dists": mean_prefix_metric(records, "dists"),
    }


def ablation_summaries():
    paths = sorted((RESULTS / "ablations").glob("*/diagnostic_evaluation.json"))
    return [ablation_summary(path) for path in paths]


def rd_summaries():
    rows = []
    paths = {
        "original_step100": ROOT / "results/pretrained_s1_test/eval/evaluation.json",
        **{stage: RESULTS / stage / "test_native1080/evaluation.json" for stage in ["d1", "d2", "d3", "d4"]},
        "extended": RESULTS / "final_evaluation/eval/evaluation.json",
    }
    for name, path in paths.items():
        report = read(path)
        if report is None:
            continue
        for row in report["budget_summary"]:
            rows.append({"model": name, **row})
    return rows


def capacity_summaries():
    rows = []
    for path in sorted((RESULTS / "capacity").glob("*/diagnostic_evaluation.json")):
        config, report = read(path.parent / "config.json"), read(path)
        records = report["records"]
        rows.append(
            {
                "experiment": path.parent.name,
                "updates": read(path.parent / "training_report.json")["completed_steps"],
                "crop": str(config["training"]["crop"]),
                "n_max": config["model"]["n_max"],
                "codebook_size": config["model"]["codebook_size"],
                "grid": str(config["model"]["grid"]),
                "base_psnr": mean_prefix_metric(records, "psnr", 0),
                "max_psnr": mean_prefix_metric(records, "psnr"),
                "deterministic_perplexity": deterministic_statistics(records)["deterministic_perplexity"],
            }
        )
    return rows


def metric_range(rows, name):
    values = [r[name] for r in rows]
    return min(values), max(values)


def measured_quality_curve(rows):
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(np.log([r["bpp"] for r in rows]), [r["psnr"] for r in rows])


def compare_native_rate(rd):
    native = read(RESULTS / "native_controls/rd_summary.json", [])
    cp = [r for r in rd if r["model"] == "extended"]
    if not native or not cp:
        return {"bd_rate": None, "reason": "missing measured rate points"}
    native.sort(key=lambda r: r["bpp"])
    cp.sort(key=lambda r: r["bpp"])
    lo, hi = (
        max(np.log(cp[0]["bpp"]), np.log(native[0]["bpp"])),
        min(np.log(cp[-1]["bpp"]), np.log(native[-1]["bpp"])),
    )
    cp_quality, native_quality = metric_range(cp, "psnr"), metric_range(native, "psnr")
    quality_lo = max(cp_quality[0], native_quality[0])
    quality_hi = min(cp_quality[1], native_quality[1])
    result = {
        "bd_rate": None,
        "common_psnr_range": [quality_lo, quality_hi],
        "reason": "BD-rate needs a common quality interval; measured quality ranges do not overlap",
        "framing": "CP application records versus native SPS/IP; overhead definitions retained",
    }
    if hi > lo:
        cp_curve, native_curve = measured_quality_curve(cp), measured_quality_curve(native)
        result.update(
            common_bpp_range=[float(np.exp(lo)), float(np.exp(hi))],
            bd_psnr=float((cp_curve.integrate(lo, hi) - native_curve.integrate(lo, hi)) / (hi - lo)),
        )
    return result


def markdown_table(rows, fields):
    if not rows:
        return "未获得该项测量，不能判为通过。"
    output = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = [
            f"{row.get(key):.5f}" if isinstance(row.get(key), float) else str(row.get(key)) for key in fields
        ]
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def plot_rd_curve(ax, rows, style, label):
    ax.plot([r["bpp"] for r in rows], [r["psnr"] for r in rows], style, label=label)


def plot_rd(ax, rd):
    for name in dict.fromkeys(r["model"] for r in rd):
        data = [r for r in rd if r["model"] == name]
        plot_rd_curve(ax, data, ".-", name)
    native = read(RESULTS / "native_controls/rd_summary.json", [])
    if native:
        plot_rd_curve(ax, native, "s--", "Native LD (SPS/IP)")
    ax.set_xscale("log")
    ax.set(xlabel="Measured bpp\n(CP records / native SPS/IP)", ylabel="1080p PSNR (dB)")
    ax.legend(fontsize=7)


def figures(ablations, rd):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    selected = [
        r
        for r in ablations
        if r["experiment"]
        in ["hard", "stochastic", "annealed", "warmup", "initialized", "balanced_vq", "balanced_decoder"]
    ]
    labels = [r["experiment"] for r in selected]
    axes[0].bar(labels, [r["deterministic_perplexity"] for r in selected])
    axes[0].set_ylabel("Deterministic perplexity\n(4-frame diagnostic)")
    axes[1].bar(labels, [r["max_psnr"] for r in selected])
    axes[1].set_ylabel("Max-prefix PSNR (overfit clip, dB)")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=60)
    plot_rd(axes[2], rd)
    fig.savefig(RESULTS / "optimization_summary.png", dpi=180)
    plt.close(fig)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def plot_diagnostic_training(ax):
    exported = []
    for name in ["continuous_4", "hard", "stochastic", "warmup", "initialized"]:
        path = RESULTS / "ablations" / name / "train.jsonl"
        rows = read_jsonl(path)
        values = []
        for row in rows:
            losses = [frame["budgets"][0]["total_loss"] for frame in row["records"]]
            values.append(float(np.mean(losses)))
            exported.append({"experiment": name, "step": row["step"], "total_loss": values[-1]})
        ax.plot([r["step"] for r in rows], values, label=name)
    ax.set(
        yscale="log",
        xlabel="Completed updates",
        ylabel="Mean training objective",
        title="Matched four-frame diagnostics",
    )
    ax.legend(fontsize=7)
    return exported


def plot_extended_training(ax):
    rows = read_jsonl(RESULTS / "multiscene_extended/train.jsonl")
    values = np.array(
        [np.mean([p["mse"] for frame in row["records"] for p in frame["budgets"]]) for row in rows]
    )
    ax.plot([r["step"] for r in rows], values, alpha=0.25, label="Actual sampled clip MSE")
    smooth = np.convolve(values, np.ones(30) / 30, mode="valid")
    ax.plot([r["step"] for r in rows][29:], smooth, label="30-update moving mean")
    ax.set(xlabel="Completed updates", ylabel="Image MSE", title="1080p multi-scene extension")
    ax.legend(fontsize=7)


def training_curves():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    exported = plot_diagnostic_training(axes[0])
    plot_extended_training(axes[1])
    fig.savefig(RESULTS / "training_curves.png", dpi=180)
    plt.close(fig)
    csv_table(RESULTS / "training_curves.csv", exported)


def main():
    ablations, rd = ablation_summaries(), rd_summaries()
    capacity = capacity_summaries()
    csv_table(RESULTS / "vq_ablations.csv", ablations)
    csv_table(RESULTS / "rd_comparison.csv", rd)
    csv_table(RESULTS / "capacity_ablations.csv", capacity)
    csv_table(RESULTS / "capacity_native1080.csv", read(RESULTS / "capacity_native1080.json", []))
    comparison = compare_native_rate(rd)
    write_json(RESULTS / "native_comparison.json", comparison)
    write_json(
        RESULTS / "summary.json",
        {
            "vq_ablations": ablations,
            "rd": rd,
            "capacity": capacity,
            "native_comparison": comparison,
            "converged": False,
        },
    )
    figures(ablations, rd)
    training_curves()
    (RESULTS / "tables.md").write_text(
        markdown_table(
            ablations,
            [
                "experiment",
                "max_psnr",
                "delta_psnr",
                "deterministic_perplexity",
                "deterministic_distinct_codes",
                "quality_drop_ratio",
            ],
        )
        + "\n\n"
        + markdown_table(rd, ["model", "tokens", "psnr", "lpips", "dists", "ms-ssim", "bpp"])
        + "\n"
    )
    print(f"Updated measured tables in {RESULTS}")


if __name__ == "__main__":
    main()
