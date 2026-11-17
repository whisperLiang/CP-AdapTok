"""Generate the experiment report exclusively from measured JSON/JSONL artifacts."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

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
    for name in ("fit", "joint", "strict_smoke"):
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
            rows.append(throughput_row(f"{budget} | {kind}", row["encoder"], row["decoder"],
                                       row["joint_sender_plus_receiver"], row["peak_allocated_mib"]))
    return "\n".join(rows)


def throughput_row(label, encoding, decoding, joint, peak_mib):
    return (f"| {label} | {encoding['fps']:.2f} | {decoding['fps']:.2f} | {joint['fps']:.2f} | "
            f"{joint['p50_ms']:.3f}/{joint['p95_ms']:.3f} | {peak_mib:.2f} |")


def token_layer_table(diagnostics):
    rows = ["| 层（K2048实际传输） | Code数 | Perplexity | 最大Code占比 |",
            "| --- | --- | --- | --- |"]
    for name, row in diagnostics["layers"].items():
        rows.append(f"| {name} | {row['used_codes']} | {row['perplexity']:.3f} | {row['most_frequent_fraction']:.3%} |")
    return "\n".join(rows)


def stream_table(data):
    rows = ["| K | 连续帧数 | Sender FPS | Receiver FPS | 联合FPS | 联合P50/P95 ms | 峰值 MiB |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for budget, measured in data["budgets"].items():
        value = measured["summary"]
        rows.append(throughput_row(f"{budget} | {value['frames']}", value["sender_latency"],
                                   value["receiver_latency"], value["sender_plus_receiver_latency"],
                                   value["peak_gpu_mib_including_metrics"]))
    return "\n".join(rows)


def component_table(data):
    budget = max(data["budgets"], key=int)
    components = data["budgets"][budget]["results"]["P"]["components_ms"]
    rows = [f"| K{budget} 首P组件 | Mean ms | P95 ms |", "| --- | --- | --- |"]
    for name, value in components.items():
        rows.append(f"| {name} | {value['mean_ms']:.3f} | {value['p95_ms']:.3f} |")
    return "\n".join(rows)


def rate_distortion_data(root):
    historical = load(root / "historical/summary.json")
    if historical is None:
        return None
    data = {"HR": [{"tokens": int(k), "bpp": v["actual_bpp"], "psnr_global": v["psnr_global"],
                    "lpips": v["lpips"], "dists": v["dists"]} for k, v in historical["budgets"].items()]}
    for branch in ("B0", "B1"):
        source = load(f"results/large_scale/formal_20261116/final_test/{branch}/validation.json")
        data[branch] = [{"tokens": r["tokens"], "bpp": r["bpp"], "psnr_global": -10 * np.log10(r["mse"]),
                         "lpips": r["lpips"], "dists": r["dists"]} for r in source["budget_summary"]]
    write_json(root / "rate_distortion.json", {"curves": data, "frames": 192,
               "scope": "same cows/goat/parkour native RGB; actual HR 13-bit+142B header vs sealed CP rANS actual bytes; different representations and reference policies; not matched numeric K",
               "official_excluded_from_rd": "no actual bitstream bpp was measured"})
    return data


def draw_rd(root, data):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, points in data.items():
        for axis, metric in zip(axes, ("psnr_global", "lpips", "dists")):
            axis.plot([p["bpp"] for p in points], [p[metric] for p in points], marker="o", label=name)
            axis.set_ylabel(metric)
            axis.set_xlabel("Actual bitstream bits / pixel")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "rate_distortion.png", dpi=160)
    plt.close(figure)


def sample_artifacts(root, destination):
    selected = [("validation_1000", "DAVIS2017_car-roundabout", 0),
                ("historical", "DAVIS2017_parkour", 31), ("final_test", "DAVIS2017_mbike-trick", 31)]
    metadata = []
    for split, video, frame in selected:
        directory = root / split / "k2048" / video
        original = directory / f"original_{frame:04d}.png"
        reconstruction = directory / f"reconstruction_{frame:04d}.png"
        if not reconstruction.exists():
            continue
        with Image.open(original) as left, Image.open(reconstruction) as right:
            panel = Image.new("RGB", (left.width * 2, left.height))
            panel.paste(left, (0, 0))
            panel.paste(right, (left.width, 0))
            target = destination / "samples" / f"{split}_{video}_{frame:04d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            panel.save(target)
        rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
        metadata.append({"path": str(target), "frame_metrics": rows[frame], "left": "original", "right": "actual ID decode"})
    write_json(root / "sample_manifest.json", metadata)
    return metadata


def deliverable_sections(root, destination):
    sections = []
    stream = load(root / "sequence_benchmark.json")
    if stream:
        sections += ["## 连续64帧独立吞吐", stream_table(stream),
                     "固定Validation car-roundabout，逐预算单遍连续64帧（2个I、62个P），预热5对I/P；双方独立状态逐帧逐位相等。每帧同步墙钟包括发送端本地解码、实际打包和接收端解码；不加载感知网络，排除读盘、H2D、指标、写盘与相等断言，无流水并行。I/P、GOP早/晚期完整计时分组见 sequence_benchmark.json。该结果与固定状态微基准均未达到联合1080p30。"]
    rd = rate_distortion_data(root)
    if rd:
        draw_rd(root, rd)
        sections += ["## 按实际bytes的率失真记录", "![实际码流率失真](../experiments/adaptok_hr/evidence/figures/rate_distortion.png)",
                     "三条曲线仅用同样192帧原生RGB。HR固定13-bit包含142-byte帧头，CP复用封存rANS实际bpp；位置/历史语义不同，不按相同K比较。官方分块无真实bpp，未放入RD图。HR在该样本上的PSNR与感知指标取舍不同，不能仅凭PSNR宣称整体更优。"]
    for sample in sample_artifacts(root, destination):
        relative = "../" + sample["path"]
        sections += [f"原图左、实际ID解码右；{sample['frame_metrics']['content_id']} 第{sample['frame_metrics']['frame']}帧，完整1080p：",
                     f"![实际重建对照]({relative})"]
    numeric = load(root / "numeric_mode_validation.json")
    if numeric:
        sections += ["## 数值模式补充核验",
                     f"固定1000步权重、同一3×8帧短Validation，在严格确定性模式下重测。相对早期短Validation，最大预算级平均MSE绝对差为 `{numeric['maximum_absolute_mse_difference']:.8g}`。该检查只覆盖这些样本，未宣称早期与新模式逐位相同；新模式的中断恢复零容差测试另列。"]
    return sections


def export_evidence(root, destination):
    destination = Path(destination)
    selected = []
    for path in sorted(root.rglob("*.json")):
        if any(p.startswith("k") and p[1:].isdigit() for p in path.relative_to(root).parts):
            continue
        selected.append(path)
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in ("training.jsonl", "frames.jsonl"):
            selected.append(path)
    selected.extend(sorted(root.glob("*.xml")))
    manifest = []
    for source in selected:
        relative = source.relative_to(root)
        manifest.append(copy_evidence(source, destination, relative))
    for source in sorted(Path("checkpoints/adaptok_hr").rglob("*.json")):
        relative = Path("checkpoints") / source.relative_to("checkpoints/adaptok_hr")
        manifest.append(copy_evidence(source, destination, relative))
    manifest.extend(media_evidence(root, destination))
    write_json(destination / "evidence_manifest.json", manifest)


def copy_evidence(source, destination, relative):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"file": str(relative), "sha256": file_hash(source), "bytes": source.stat().st_size}


def media_evidence(root, destination):
    for name in ("training_curves.png", "rate_distortion.png"):
        if (root / name).exists():
            copy_evidence(root / name, destination, Path("figures") / name)
    paths = sorted(destination.rglob("*.png")) + sorted(destination.rglob("*.bin"))
    return [{"file": str(p.relative_to(destination)), "sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in paths]


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
    sections += ["![训练曲线](../experiments/adaptok_hr/evidence/figures/training_curves.png)"]
    token_diagnostics = load(root / "token_diagnostics.json")
    if token_diagnostics:
        sections += ["## 已定位的表示瓶颈", token_layer_table(token_diagnostics),
                     "该统计来自完整Validation中K2048实际码流，Fine层在这些样本上没有传递内容变化。额外同状态前缀重建见 token_diagnostics.json；辅助分支不更新参考。不能把独立预算闭环的差值全部当作当前帧量化效应。"]
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
                     "I/首P各5预热、20重复，batch1。联合计算包含发送端本地解码与独立接收端，不能把Encoder FPS直接称端到端FPS；该表为固定状态微基准，连续视频另测。",
                     f"参数量 `{exemplar['flops']['parameters']}`；I/P GFLOPs：" +
                     ", ".join(f"{k}={v['gflops']:.4f}" for k, v in exemplar["flops"]["frames"].items()) + ". " + exemplar["flops"]["convention"] + "。FLOPs口径是一次Encoder+VQ+一次Decoder前向；不含接收端的额外一次Decoder，联合速度则实际计入这次解码。",
                     "Conv、window attention、token/VQ、lookup/scatter、低分辨率补全、高分辨率上采样、状态、13-bit packing/unpacking均独立保存；分量探测含同步开销，不相加冒充整机性能。", component_table(performance)]
    sections += ["## 封存Baseline", baseline_table(base["rows"]),
                 "官方K为每Tile每16帧，HR为每完整帧。无重叠K2048约17280 Token/源帧，重叠约28160；不按相同K宣称同码率公平。官方 bpp=null；HR是真实固定13-bit流，包含帧头与CRC，没有熵模型。",
                 "## 继续训练决策与未完成项目", f"```json\n{json.dumps(decision, ensure_ascii=False, indent=2)}\n```",
                 "Conv-only / local attention / temporal、fixed/nested可通过配置运行匹配消融；未运行的消融不产生质量数字。Scorer支持冻结Tokenizer与同状态真实ID标签，只有稳定性门槛通过后才进行正式训练，不把单元测试称为Scorer实验。",
                 "## 验收与证据", f"```json\n{json.dumps(load(root / 'acceptance.json', {}), ensure_ascii=False, indent=2)}\n```",
                 "测试覆盖、复杂度/重复审计和行为变异结果见 evidence；crapper/dryer/mutator专用工具未安装，替代工具明确记录。源数据与大Checkpoint保存在本地忽略目录，身份、配置、完整训练JSONL与小型结果证据随Git提交。",
                 "所有质量指标在原生float32 RGB计算，样图显示量化不参与指标。单训练种子、小数据/短程Pilot，不据此宣称训练已收敛、同码率领先或1080p30部署完成。"]
    draw_curves(root)
    sections.extend(deliverable_sections(root, args.export))
    args.output.write_text("\n\n".join(sections) + "\n")
    write_json(root / "baseline_comparison.json", {"source": "results/original_adaptok_1080p/absolute_quality_comparison.json",
                "source_sha256": file_hash("results/original_adaptok_1080p/absolute_quality_comparison.json"),
                "rows": base["rows"], "scope": "absolute quality; token units and bitstream differ"})
    export_evidence(root, args.export)
    print(str(args.output))


if __name__ == "__main__":
    main()
