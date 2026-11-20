"""Publish figures and a status report using only recorded measurements."""

import csv
import json
from collections import defaultdict
from statistics import mean

from aip_vtok.diagnostics.diag0.io import command, json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO

from .evaluation import BASELINES, CLOSED, JOINT, VIZ_BUDGETS
from .protocol import BASE, BRANCH, OUTPUT
from .scan import BUDGETS


def optional_json(path):
    return json_at(path) if path.is_file() else {}


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def architecture(plt, directory):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set(xlim=(0, 13), ylim=(0, 4.5))
    ax.axis("off")
    rows = [
        (3.35, ["1080p RGB", "Frozen DCVC I/P", "Orthogonal I/P\n128 → 48D",
                "8 × 6D A2 FSQ\nShared hard scan", "Integer rANS\n≤ 128 tokens/packet"]),
        (1.95, ["1080p RGB", "Frozen DCVC decoder", "Transpose transform",
                "Two-scale SLD\n128D + hard preserve", "Sparse 48D canvas\nVerified integer prefix"]),
    ]
    colors = ("#e8eff2", "#d3e9ed")
    for index, (y, labels) in enumerate(rows):
        for i, label in enumerate(labels):
            x = 1.3 + 2.6 * i
            ax.text(x, y, label, ha="center", va="center", color="#162a33",
                    bbox={"boxstyle": "round,pad=0.4", "fc": colors[index], "ec": "#597680"})
            if i < len(labels) - 1:
                left, right = x + 0.95, x + 1.65
                a, b = (left, right) if index == 0 else (right, left)
                ax.annotate("", (b, y), (a, y), arrowprops={"arrowstyle": "->", "color": "#355a62"})
    ax.annotate("", (11.7, 2.30), (11.7, 2.98), arrowprops={"arrowstyle": "->", "color": "#355a62"})
    ax.annotate("I2048 / P510 → LatentToRef → next-frame context", (3.9, 0.95),
                (8.0, 0.8), va="top", ha="center", arrowprops={"arrowstyle": "->", "color": "#a14921"})
    ax.set_title("AIP-VTok Joint V1", loc="left", fontsize=17)
    fig.savefig(directory / "figure_1_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rd_figures(plt, directory, trajectories):
    for i, (metric, label) in enumerate((("global_psnr", "Global PSNR (dB)"), ("lpips", "LPIPS"),
                                       ("dists", "DISTS")), 2):
        fig, ax = plt.subplots(figsize=(7, 4))
        for seed, rows in trajectories.items():
            x, y = zip(*[(r["actual_bpp"], r[metric]) for r in rows])
            ax.plot(x, y, marker="o", label=f"Joint V1, seed {seed}")
        ax.set(xlabel="Actual bpp including reference prefixes and packet headers", ylabel=label, xscale="log")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(directory / f"figure_{i}_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def reconstruction_figure(directory, crops):
    from PIL import Image, ImageDraw, ImageFont

    labels = ("Original", *BASELINES, JOINT)
    panel_w, panel_h = 384, 246
    image = Image.new("RGB", (panel_w * len(labels), panel_h * len(VIZ_BUDGETS)), "#182227")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for i, k in enumerate(VIZ_BUDGETS):
        for j, label in enumerate(labels):
            x, y = j * panel_w, i * panel_h
            draw.text((x + 8, y + 7), f"{label} • K{k}", fill="white", font=font)
            crop = crops / f"{label}_{k}.png"
            if crop.is_file():
                with Image.open(crop) as source:
                    image.paste(source, (x, y + 30))
            elif label in BASELINES and k > 8192:
                draw.text((x + 165, y + 115), "N/A", fill="white", font=font)
            else:
                raise RuntimeError(f"missing reconstruction crop {crop}")
    image.save(directory / "figure_5_reconstructions.png")


def temporal_figures(plt, directory, rows, benchmark):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for budget in VIZ_BUDGETS:
        points = defaultdict(list)
        for row in rows:
            if int(row["budget"]) == budget:
                points[int(row["gop_position"])].append(row)
        steps = sorted(points)
        axes[0].plot(steps, [mean(float(v["psnr"]) for v in points[i]) for i in steps], label=f"K{budget}")
    reference = [r for r in rows if int(r["budget"]) == 510 and r["video"] == HORIZON_VIDEO]
    axes[1].plot([int(r["frame"]) % 32 for r in reference[:32]],
                 [float(r["reference_latent_rms"]) for r in reference[:32]], label="GOP 0")
    for start in range(32, len(reference), 32):
        group = reference[start:start + 32]
        axes[1].plot([int(r["frame"]) % 32 for r in group], [float(r["reference_latent_rms"]) for r in group],
                     label=f"GOP {start // 32}")
    axes[0].set_ylabel("Mean-frame PSNR (dB)")
    axes[1].set(xlabel="GOP position", ylabel="Reference latent RMS")
    for ax in axes:
        ax.legend(ncol=3)
        ax.grid(alpha=0.2)
    fig.savefig(directory / "figure_6_gop_quality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    keys = ("encoder_ms", "transform_ms", "fsq_ms", "scan_ms", "entropy_encode_ms", "entropy_decode_ms",
            "sld_ms", "rgb_decode_ms", "tx_reference_ms", "rx_reference_ms", "tx_context_ms", "rx_context_ms",
            "ids_transfer_ms")
    names = ("DCVC encoder", "Transform", "FSQ", "Scan", "Entropy encode", "Entropy decode",
             "SLD recovery", "RGB decode", "TX reference", "RX reference", "TX context", "RX context", "ID transfer")
    groups = {k: [r for r in benchmark if int(r["budget"]) == k] for k in BUDGETS}
    fig, ax = plt.subplots(figsize=(10, 5))
    left = [0.0] * len(BUDGETS)
    for name, key in zip(names, keys, strict=True):
        series = [mean(float(r[key]) for r in points) for points in groups.values()]
        ax.barh([str(k) for k in BUDGETS], series, left=left, label=name)
        left = [a + b for a, b in zip(left, series, strict=True)]
    ax.set(xlabel="Mean synchronized elapsed time (ms)", ylabel="Display tokens")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(directory / "figure_7_latency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_figure(plt, directory):
    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    for seed in (1, 2):
        source = OUTPUT / f"seed{seed}"
        journal = source / "updates.jsonl"
        if not journal.is_file():
            continue
        steps, losses = [], []
        with journal.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["step"] % 25 == 0:
                    steps.append(record["step"])
                    losses.append(mean(sum(sum(b["weighted"].values()) for b in f["budgets"].values())
                                       for f in record["frames"]))
        axes[0].plot(steps, losses, alpha=0.6, label=f"Seed {seed}")
        checkpoints = sorted((source / "periodic").glob("step_*/evaluation.json"))
        xs = [int(q.parent.name.split("_")[1]) for q in checkpoints]
        summaries = [json_at(p)["summary"] for p in checkpoints]
        for ax, key, title in zip(axes[1:], ("global_psnr", "lpips", "dists"),
                                  ("PSNR (dB)", "LPIPS", "DISTS"), strict=True):
            ax.plot(xs, [mean(r[key] for r in s) for s in summaries], label=f"Seed {seed}")
            ax.set_ylabel(title)
    axes[0].set_ylabel("Weighted train loss")
    axes[-1].set_xlabel("Optimizer updates")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(directory / "figure_8_training_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    architecture(plt, directory)
    root = OUTPUT / "seed1"
    validation = root / "validation"
    if (validation / "evaluation.json").is_file():
        trajectories = {}
        for seed in (1, 2):
            data = optional_json(OUTPUT / f"seed{seed}/validation/evaluation.json")
            if data:
                trajectories[seed] = [r for r in data["summary"] if r["method"] == JOINT and r["history"] == CLOSED]
        rd_figures(plt, directory, trajectories)
        reconstruction_figure(directory, validation / "crops")
        benchmark_path = root / "benchmark" / "validation_per_frame.csv"
        if benchmark_path.is_file():
            rows = [r for r in csv_rows(validation / "validation_per_frame.csv")
                    if r["method"] == JOINT and r["history"] == CLOSED]
            temporal_figures(plt, directory, rows, csv_rows(benchmark_path))
        training_figure(plt, directory)


def metric_table(rows):
    lines = ["| 方法 | K | 全局 PSNR | 帧均 PSNR | LPIPS | DISTS | 实际 bpp | Raw bpp | 端到端 p50/p95 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["history"] != CLOSED:
            continue
        if r["status"] == "N/A":
            lines.append(f'| {r["method"]} | {r["budget"]} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
            continue
        actual = f'{r["actual_bpp"]:.6f}' if "actual_bpp" in r else "N/A"
        lines.append(f'| {r["method"]} | {r["budget"]} | {r["global_psnr"]:.3f} | {r["mean_frame_psnr"]:.3f} | '
                     f'{r["lpips"]:.5f} | {r["dists"]:.5f} | {actual} | {r["raw_bpp"]:.6f} | '
                     f'{r["end_to_end_ms_p50_ms"]:.2f}/{r["end_to_end_ms_p95_ms"]:.2f} |')
    return lines


def write_report():
    report = [
        "# AIP-VTok Joint V1 实验报告", "", f"- 源提交 {BASE}",
        f'- 分支 {BRANCH}，生成报告时的代码 HEAD {command("git", "rev-parse", "HEAD")}',
        "- IMPLEMENTED。架构和可见信息边界见 [architecture_contract.md](architecture_contract.md)。",
        "- I/P 的独立 128D 正交变换与 A2 FSQ 产生 8 组 × 6D 的 16-bit 可表示整数符号，"
        "共享扫描支持八个前缀。高、低两尺度 SLD 在变换域保持真实观察值。",
        "- 单包上限 128 Token，各自维护 32-bit rANS 状态，使用 12-bit 整数 CDF 和 CRC32。"
        "I2048/P510 的固定参考前缀计入实际传输，分包和头部计费见 rate_ledger.csv。",
        "- 训练配置见 [joint_v1.yaml](../../../configs/aip_vtok/joint_v1.yaml)。每帧预算去重等权，"
        "参考项按帧计一次。学习率使用 1,000 步 warmup 和余弦衰减至峰值的 10%。"
        "A/B 窗口 4 帧，C 窗口 8 帧。全部训练保留真实 GOP 相位和当期模型的无梯度 burn-in。",
        "", "## 工程验证与训练", "",
    ]
    engineering = optional_json(OUTPUT / "engineering_tests.json")
    if engineering:
        report.append(f'- TESTED。工程证据见 [engineering_tests.json](engineering_tests.json)，通过 {engineering["passed"]} 项。')
    for seed in (1, 2):
        status = optional_json(OUTPUT / f"train_seed{seed}.json")
        if not status:
            report.append(f"- Seed {seed} INCOMPLETE，尚无正式 checkpoint。")
            continue
        checkpoint = status["checkpoint"]
        report.append(f'- Seed {seed} {status["status"]}，已保存 {status["steps"]:,}/{status["target"]:,} 步，'
                      f'checkpoint {status["checkpoint_directory"]}/{checkpoint["file"]}，SHA256 {checkpoint["sha256"]}。')
        report.append(f'  扫描版本 {checkpoint["scan"]["scan_version"]}，'
                      f'SHA256 {checkpoint["scan"]["scan_sha256"]}。')
        report.append(f'  各阶段有效训练秒数 {status["stage_seconds"]}，'
                      f'H100 峰值分配 {status["peak_gpu_mib"]:.0f} MiB，Test 像素读取 {status["pixel_reads"]["test"]}。')
        if seed == 1:
            counts = status["parameters"]
            report.append(f"  总参数 {sum(counts.values()):,}，按模块 {counts}。")
        selection = optional_json(OUTPUT / f"seed{seed}/selection.json")
        if selection:
            report.append(f'  冻结 Validation 模型选择为 step {selection["chosen"]["step"]}。')
    for split, label in (("validation", "VALIDATED"), ("test", "TESTED_ON_HELDOUT")):
        report.extend(("", f"## {split}", ""))
        path = OUTPUT / f"seed1/{split}/evaluation.json"
        if path.is_file():
            report.extend([label, "", *metric_table(json_at(path)["summary"])])
        else:
            report.append("INCOMPLETE。正式测量尚未产生。")
    report += [
        "", "旧方案的高预算标为 N/A。旧方案的 raw 或估计载荷采用原协议计量。"
        "逐帧的参考指纹、后段质量、额外长序列结果和同步计时保存在相应评估目录。",
        "", "## 复现", "", "```bash",
        "python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --train",
        "python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --resume",
        "python scripts/aip_vtok/evaluate_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --split validation",
        "python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 2 --train",
        "python scripts/aip_vtok/evaluate_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --split test",
        "python scripts/aip_vtok/benchmark_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1", "```", "",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / "final_report.md.tmp"
    temporary.write_text("\n".join(report), encoding="utf-8")
    temporary.replace(OUTPUT / "final_report.md")
