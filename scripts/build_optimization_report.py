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
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ablation_summaries():
    summaries = []
    for path in sorted((RESULTS / "ablations").glob("*/diagnostic_evaluation.json")):
        report = read(path)
        records = report["records"]
        counts = np.sum([r["vq"]["deterministic"]["counts"] for r in records], axis=0)
        deltas = [r["prefixes"][-1]["psnr"] - r["prefixes"][0]["psnr"] for r in records]
        adjacent = [(a["mse"], b["mse"]) for r in records for a, b in zip(r["prefixes"], r["prefixes"][1:])]
        summaries.append(
            {
                "experiment": path.parent.name,
                "updates": 100,
                "frames_per_update": len(records),
                "continuous_diagnostic": report["continuous_only_diagnostic"],
                "base_psnr": float(np.mean([r["prefixes"][0]["psnr"] for r in records])),
                "max_psnr": float(np.mean([r["prefixes"][-1]["psnr"] for r in records])),
                "delta_psnr": float(np.mean(deltas)),
                "deterministic_perplexity": float(
                    np.mean([r["vq"]["deterministic"]["perplexity"] for r in records])
                ),
                "deterministic_distinct_codes": int((counts > 0).sum()),
                "quality_drop_ratio": float(np.mean([b > a for a, b in adjacent])),
                "max_lpips": float(np.mean([r["prefixes"][-1]["lpips"] for r in records])),
                "max_dists": float(np.mean([r["prefixes"][-1]["dists"] for r in records])),
            }
        )
    return summaries


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
    axes[0].set_ylabel("Deterministic perplexity (4-frame diagnostic)")
    axes[1].bar(labels, [r["max_psnr"] for r in selected])
    axes[1].set_ylabel("Max-prefix PSNR (overfit clip, dB)")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=60)
    for name in dict.fromkeys(r["model"] for r in rd):
        data = [r for r in rd if r["model"] == name]
        axes[2].plot([r["bpp"] for r in data], [r["psnr"] for r in data], ".-", label=name)
    axes[2].set(xlabel="Actual application bpp", ylabel="1080p PSNR (dB)")
    axes[2].legend(fontsize=7)
    fig.savefig(RESULTS / "optimization_summary.png", dpi=180)
    plt.close(fig)


def main():
    ablations, rd = ablation_summaries(), rd_summaries()
    csv_table(RESULTS / "vq_ablations.csv", ablations)
    csv_table(RESULTS / "rd_comparison.csv", rd)
    write_json(RESULTS / "summary.json", {"vq_ablations": ablations, "rd": rd, "converged": False})
    figures(ablations, rd)
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
