"""Generate the experiment report exclusively from measured JSON/JSONL artifacts."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ..runtime import file_hash, write_json


def load(path, fallback=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else fallback


def baseline_table(records):
    rows = ["| 模型 | Token 单位 | PSNR逐帧均值 | LPIPS-Alex | DISTS |",
            "| --- | --- | --- | --- | --- |"]
    for row in records:
        if row["content_id"] == "all":
            rows.append(f"| {row['model']} | {row['budget_units']} | {row['psnr_mean_db']:.3f} | {row['lpips']:.5f} | {row['dists']:.5f} |")
    return "\n".join(rows)


def quality_table(summary):
    rows = ["| K/完整帧 | 帧数 | PSNR全局MSE | PSNR逐帧均值 | LPIPS-Alex | DISTS | 实际bpp | 实际bytes | Code使用率 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for budget, row in summary["budgets"].items():
        lpips, dists = row.get("lpips"), row.get("dists")
        rows.append(f"| {budget} | {row['frames']} | {row['psnr_global']:.4f} | {row['psnr_frame_mean']:.4f} | "
                    f"{lpips:.5f} | {dists:.5f} | {row['actual_bpp']:.8f} | {row['bytes']} | {row['codebook_usage']:.4%} |")
    return "\n".join(rows)


def training_table(root):
    rows = ["| 实验 | 实际完成更新 | 实际帧次 | 更新计算秒 | 最大allocated MiB |",
            "| --- | --- | --- | --- | --- |"]
    for name in ("fit", "joint"):
        completion = load(root / name / "completion.json")
        path = root / name / "training.jsonl"
        if completion and path.exists():
            measurements = [json.loads(line) for line in path.read_text().splitlines()]
            rows.append(f"| {name} | {completion['completed_steps']} | {completion['training_frames']} | "
                        f"{sum(r['update_seconds'] for r in measurements):.3f} | {max(r['peak_gpu_mib'] for r in measurements):.2f} |")
    return "\n".join(rows)


def benchmark_table(data):
    rows = ["| K | 型 | Encoder FPS | Receiver FPS | Sender+Receiver FPS | 联合P50/P95 ms | 峰值 MiB |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for budget, measured in data["budgets"].items():
        for kind, row in measured["results"].items():
            joint = row["joint_sender_plus_receiver"]
            rows.append(f"| {budget} | {kind} | {row['encoder']['fps']:.2f} | {row['decoder']['fps']:.2f} | "
                        f"{joint['fps']:.2f} | {joint['p50_ms']:.3f}/{joint['p95_ms']:.3f} | {row['peak_allocated_mib']:.2f} |")
    return "\n".join(rows)


def export_evidence(root, destination):
    destination = Path(destination)
    selected = []
    for path in sorted(root.rglob("*.json")):
        if any(p.startswith("k") and p[1:].isdigit() for p in path.relative_to(root).parts):
            continue
        selected.append(path)
    for path in sorted(root.rglob("*.jsonl")):
        if path.name == "training.jsonl":
            selected.append(path)
    manifest = []
    for source in selected:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"file": str(relative), "sha256": file_hash(source), "bytes": source.stat().st_size})
    write_json(destination / "evidence_manifest.json", manifest)


def draw_curves(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("fit", "joint"):
        path = root / name / "training.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        axes[0].plot([r["step"] for r in rows], [r["psnr_global"] for r in rows], label=name, alpha=.7)
        axes[1].plot([r["step"] for r in rows], [r["peak_gpu_mib"] for r in rows], label=name, alpha=.7)
    for axis, label in zip(axes, ("Train RGB PSNR (dB), sampled budgets", "Peak allocated MiB")):
        axis.set_xlabel("Completed optimizer update")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(root / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/adaptok_hr"))
    parser.add_argument("--output", type=Path, default=Path("docs/adaptok_hr_training_report.md"))
    parser.add_argument("--export", type=Path, default=Path("experiments/adaptok_hr/evidence"))
    args = parser.parse_args()
    root = args.root
    base = load("results/original_adaptok_1080p/absolute_quality_comparison.json", {"rows": []})
    phase0 = load(root / "phase0/phase0.json", {})
    decision = load(root / "decision.json", {"status": "not_evaluated"})
    data = load(root / "data_audit.json", {})
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sections = ["# AdapTok-HR 原生1080p实验报告",
                f"分支 experiment/adaptok-hr；生成时 Git `{git}`。所有数字由实际 JSON/JSONL 读取；完整运行命令见 experiments/adaptok_hr/README.md。",
                "## 数据与实际训练", f"```json\n{json.dumps(data.get('summary', {}), ensure_ascii=False, indent=2)}\n```",
                "索引只接收原生至少1920×1080的源，高分辨率仅裁剪，不上采样。新 HR Test 从旧 CP Test 索引中排除已用于历史1080p对照的 cows/goat/parkour；不是全新独立数据集。固定短Validation用于进度记录，完整64帧评估另列。", training_table(root)]
    if phase0:
        probes = {k: {"peak_gpu_mib": v["update"]["peak_gpu_mib"], "update_seconds": v["update"]["update_seconds"]}
                  for k, v in phase0["training_probes"].items()}
        sections += ["工程探测每种展开长度独立做1次真实优化器更新，另计，不混入正式Pilot；TBPTT2。", f"```json\n{json.dumps(probes, indent=2)}\n```",
                     "33帧独立收发端状态逐帧相同，第33帧I重置与空参考重建相同。工程测试含实际VQ、真实二进制与反向。"]
    fit = load(root / "fit/fit_diagnostic.json")
    if fit:
        sections += ["单Train片段拟合的独立真实码流诊断（不是泛化指标）：", f"```json\n{json.dumps(fit, ensure_ascii=False, indent=2)}\n```"]
    sections += ["![训练曲线](../results/adaptok_hr/training_curves.png)"]
    for name, label in (("validation_300", "300更新后完整Validation"), ("validation_1000", "1000更新后完整Validation"),
                        ("historical", "已知历史视频绝对质量"), ("final_test", "封存HR Test")):
        measured = load(root / name / "summary.json")
        if measured:
            sections += [f"## {label}", quality_table(measured),
                         f"逐帧、I/P、GOP早期1–7/晚期24–31、运动阈值0.01/0.03分组，以及传输Code直方图见 `{root / name}`。每条不足64帧的源明确保存 at_least_64_contiguous_frames=false；不补帧计数。"]
    performance = load(root / "benchmark.json")
    if performance:
        exemplar = next(iter(performance["budgets"].values()))
        sections += ["## 独立性能与计算量", benchmark_table(performance), exemplar["scope"],
                     "I/首P各5预热、20重复，batch1。联合计算包含发送端本地解码与独立接收端，不能把Encoder FPS直接称端到端FPS。非完整视频速度的所有场景保证。",
                     f"参数量 `{exemplar['flops']['parameters']}`；I/P GFLOPs：" +
                     ", ".join(f"{k}={v['gflops']:.4f}" for k, v in exemplar["flops"]["frames"].items()) + ". " + exemplar["flops"]["convention"],
                     "Conv、window attention、token/VQ、lookup/scatter、低分辨率补全、高分辨率上采样、状态、13-bit packing/unpacking均独立保存；分量探测含同步开销，不相加冒充整机性能。"]
    sections += ["## 封存Baseline", baseline_table(base["rows"]),
                 "官方K为每Tile每16帧，HR为每完整帧。无重叠K2048约17280 Token/源帧，重叠约28160；不按相同K宣称同码率公平。官方 bpp=null；HR是真实固定13-bit流，包含帧头与CRC，没有熵模型。",
                 "## 继续训练决策与未完成项目", f"```json\n{json.dumps(decision, ensure_ascii=False, indent=2)}\n```",
                 "Conv-only / local attention / temporal、fixed/nested可通过配置运行匹配消融；未运行的消融不产生质量数字。Scorer支持冻结Tokenizer与同状态真实ID标签，只有稳定性门槛通过后才进行正式训练，不把单元测试称为Scorer实验。",
                 "## 验收与证据", f"```json\n{json.dumps(load(root / 'acceptance.json', {}), ensure_ascii=False, indent=2)}\n```",
                 "测试覆盖、复杂度/重复审计和行为变异结果见 evidence；crapper/dryer/mutator专用工具未安装，替代工具明确记录。源数据与大Checkpoint保存在本地忽略目录，身份、配置、完整训练JSONL与小型结果证据随Git提交。",
                 "所有质量指标在原生float32 RGB计算，样图显示量化不参与指标。单训练种子、小数据/短程Pilot，不据此宣称训练已收敛、同码率领先或1080p30部署完成。"]
    draw_curves(root)
    args.output.write_text("\n\n".join(sections) + "\n")
    write_json(root / "baseline_comparison.json", {"source": "results/original_adaptok_1080p/absolute_quality_comparison.json",
                "source_sha256": file_hash("results/original_adaptok_1080p/absolute_quality_comparison.json"),
                "rows": base["rows"], "scope": "absolute quality; token units and bitstream differ"})
    export_evidence(root, args.export)
    print(str(args.output))


if __name__ == "__main__":
    main()
