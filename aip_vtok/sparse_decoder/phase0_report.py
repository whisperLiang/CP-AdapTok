"""Publish the physical-unit comparisons and paired K510 curves."""

import matplotlib

from .evaluation import CLOSED, FIXED
from .phase0 import FACTORS
from .protocol import OUTPUT

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LABELS = {"scan_a1": "Q2 − Q1 · A1 扫描效应", "scan_a2": "Q4 − Q3 · A2 扫描效应",
          "quant_old": "Q3 − Q1 · C1 量化效应", "quant_sp1": "Q4 − Q2 · SP-1 量化效应",
          "interaction": "扫描 × 量化交互", "Q4_minus_Q1": "Q4 − Q1 · 联合差距"}


def table(headers, rows):
    def line(items):
        return "| " + " | ".join(map(str, items)) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def quality(m):
    return (f'{m["psnr_global_mse"]:.3f}', f'{m["psnr_frame_mean"]:.3f}',
            f'{m["lpips"]:.5f}', f'{m["dists"]:.5f}')


def effects_table(entries):
    for subgroup, effects in entries.items():
        for history in (FIXED, CLOSED, "closed_minus_fixed"):
            for key, value in effects[history].items():
                yield subgroup, history, LABELS[key], *quality(value)


def plot_curves(rows, counts):
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True, layout="constrained")
    offsets = dict(zip(counts, (sum(list(counts.values())[:i]) for i in range(len(counts)))))
    for col, history in enumerate((FIXED, CLOSED)):
        for row, (metric, label) in enumerate((("psnr", "PSNR (dB)"), ("lpips", "LPIPS"), ("dists", "DISTS"))):
            ax = axes[row, col]
            for q in FACTORS:
                subset = [r for r in rows if r["history"] == history and r["arm"] == q]
                ax.plot([offsets[r["video"]] + r["frame"] for r in subset], [r[metric] for r in subset],
                        linewidth=1, label=q)
            for start in tuple(offsets.values())[1:]:
                ax.axvline(start - 0.5, color="0.6", linewidth=0.6)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.22)
        axes[0, col].set_title(("Shared FIXED history", "Independent CLOSED history")[col])
        axes[0, col].legend(ncol=4, fontsize=8)
        axes[-1, col].set_xticks([offsets[v] + counts[v] // 2 for v in counts],
                                [v.split(":")[-1] for v in counts])
    fig.savefig(OUTPUT / "phase0_k510.png", dpi=165)
    plt.close(fig)


def publish(diag, factorial, rows):
    headers = ("全局 PSNR dB", "帧均 PSNR dB", "LPIPS", "DISTS")
    points = diag["points"]
    comparisons = diag["comparison_to_a8474ee"]
    scan = factorial["validation"]["contrasts"]["all"]
    fixed = scan[FIXED]
    feedback = scan["closed_minus_fixed"]["Q4_minus_Q1"]
    diagnostics = table(("方案", "K", "帧型", *headers, "潜在 MSE"),
                        ((p["arm"], p["budget"] or "连续", kind, *quality(m), f'{m["latent_mse"]:.6f}')
                         for p in points for kind, m in ((k, p[k]) for k in ("all", "I", "P"))))
    changes = table(("对照", "帧型", *headers),
                    ((step, kind, *quality(m)) for step, groups in diag["changes"].items()
                     for kind, m in groups.items()))
    history = table(("本轮", "旧方案", *headers, "I 帧最大 RGB MSE 差"),
                    ((arm, d["old_arm"], *quality(d["old"]["all"]), f'{d["max_intra_frame_mse_abs_difference"]:.2g}')
                     for arm, d in comparisons.items()))
    lines = [
        "# Phase 0 · 无训练信息瓶颈诊断",
        "原生 1080p、174 帧 Validation，A2/SP-1 与 C1 扫描均来自冻结 Train 资产。"
        "FIXED 全部使用 C1 A1/B2 的 I2048/P510 历史，CLOSED 按各组自身的码流推进参考。",
        "## 通道、量化与空间采样",
        "O0 解码完整 128D aligned latent。O1 保留未量化的 24D 并补零。"
        "O2 对同一 24D 的全部 32640 个符号做 A2 FSQ。O3 是 SP-1 与 σ=1.5 高斯。O4 加载原 W1。",
        diagnostics,
        "四项差值按后者减前者计算。空间采样与学习恢复采用 K8192。"
        "这些差值用于配对诊断。O0 衡量冻结解码器在完整对齐潜在输入下的表现。",
        changes,
        "## 与 a8474ee 的稠密对照",
        "旧 A0/A2 和当前 O1/O2 使用相同 Validation 帧及全局 MSE 聚合。"
        "旧 P 帧参考来自稠密 A1，当前 P 帧参考来自稀疏 C1 前缀。整体数值不宜作为直接复现，"
        "I 帧没有此前 GOP 历史，表中给出其逐帧核验结果。",
        history,
        "## K510 量化器 × 扫描表",
        "各单元使用同坐标同通道，每帧传输 510 个匹配自身量化器的 FSQ 符号。"
        "交互项为 (Q4 − Q3) − (Q2 − Q1)。CLOSED − FIXED 表示引入独立参考反馈后差值的变化。",
        f'在共享历史下，SP-1 的 PSNR 效应为 A1 {fixed["scan_a1"]["psnr_global_mse"]:+.3f} dB、'
        f'A2 {fixed["scan_a2"]["psnr_global_mse"]:+.3f} dB。同一扫描下 A2 的 PSNR 效应为 '
        f'{fixed["quant_old"]["psnr_global_mse"]:+.3f} 至 {fixed["quant_sp1"]["psnr_global_mse"]:+.3f} dB。'
        f'独立历史使 Q4 相对 Q1 的 PSNR 进一步变化 {feedback["psnr_global_mse"]:+.3f} dB。'
        "扫描对 PSNR 的影响最大，同时降低了 LPIPS 和 DISTS。量化和交互在总体感知指标上的效应较小。",
        "![K510 逐帧重建质量](phase0_k510.png)",
    ]
    for scope in ("validation", "horizon", "late"):
        subset = factorial[scope]
        measurements = ((p["history"], p["arm"], k, p[k]["frames"], *quality(p[k]))
                        for p in subset["points"] for k in ("all", "I", "P") if k in p)
        lines.extend([
            f"### {scope}",
            table(("历史", "方案", "帧型", "帧数", *headers), measurements),
            table(("帧型", "历史", "效应", *headers), effects_table(subset["contrasts"])),
        ])
    lines.extend([
        "## 复现",
        "\x60python scripts/aip_vtok/run_sparse_decoder.py phase0\x60",
        f'输入清单 SHA256 为 \x60{diag["provenance_sha256"]}\x60。结果见'
        " [资产清单](provenance.json) · [逐帧 CSV](phase0_per_frame.csv) ·"
        " [诊断 JSON](phase0_diagnostics.json) · [归因 JSON](phase0_factorial.json)。",
    ])
    (OUTPUT / "phase0_report.md").write_text("\n\n".join(lines) + "\n")
    plot_curves(rows, diag["cohort"])
