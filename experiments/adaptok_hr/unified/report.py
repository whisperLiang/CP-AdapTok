"""Render the final report directly from real matched study evidence."""

import json
from datetime import datetime

import numpy as np

from .evaluate import read_rows
from .run import PACKAGE, ROOT


def read(path):
    return json.loads(path.read_text())


def measurements():
    return [(p.parents[1].name, p.parent, read(p)) for p in sorted(ROOT.glob("S*/validation_*_64/summary.json"))]


def quality_table(lines, values):
    lines.extend(["", "| 模型 | 步 | K | bytes/帧 | bpp | PSNR-global | PSNR-frame-mean | LPIPS-Alex | DISTS | Cb MAE | Cr MAE | 色差方差保留率 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, _, result in values:
        for k, q in result["closed_loop"].items():
            lines.append(f"| {name} | {result['completed_steps']} | {k} | {q['bytes']/q['frames']:.0f} | {q['actual_bpp']:.8f} | "
                         f"{q['psnr_global']:.4f} | {q['psnr_frame_mean']:.4f} | {q['lpips']:.6f} | {q['dists']:.6f} | "
                         f"{q['cb_mae']:.6f} | {q['cr_mae']:.6f} | {q['color_difference_variance_retention']:.4f} |")


def groups_table(lines, values):
    lines.extend(["", "| 模型/步 | K | 帧段 | 帧数 | bytes/帧 | bpp | PSNR-global | PSNR-frame-mean | LPIPS | DISTS | Cb MAE | Cr MAE | 色差方差保留率 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, _, result in values:
        for k, summary in result["closed_loop"].items():
            for group in ("I", "P", "late_GOP"):
                row = summary[group]
                lines.append(f"| {name}/{result['completed_steps']} | {k} | {group} | {row['frames']} | {row['bytes']/row['frames']:.0f} | {row['actual_bpp']:.8f} | {row['psnr_global']:.4f} | "
                             f"{row['psnr_frame_mean']:.4f} | {row['lpips']:.6f} | {row['dists']:.6f} | "
                             f"{row['cb_mae']:.6f} | {row['cr_mae']:.6f} | {row['color_difference_variance_retention']:.4f} |")


def fine_table(lines, values):
    lines.extend(["", "| 模型/步 | 同状态 K2048−K1024 PSNR | 同状态 LPIPS 改善 | Fine 置乱 ΔMSE | Fine 跨视频 ΔMSE |",
                  "|---|---:|---:|---:|---:|"])
    for name, _, result in values:
        fine = result["full_fine"]
        lines.append(f"| {name}/{result['completed_steps']} | {result['same_state_gain_db']:+.5f} | "
                     f"{result['same_state_lpips_gain']:+.6f} | {fine['shuffle']['mse_increase']:+.8f} | {fine['cross']['mse_increase']:+.8f} |")


def resources_table(lines):
    lines.extend(["", "| 模型 | 新增更新 | 新增训练帧 | 更新总秒数 | 平均/P95 秒/步 | 训练窗口秒数 | 峰值 MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for path in sorted(ROOT.glob("S*/completion.json")):
        name = path.parent.name
        start = 1300 if "_U" in name else 0
        rows = [r for r in read_rows(path.parent / "training.jsonl") if r["step"] > start]
        times = [r["update_seconds"] for r in rows]
        run = read(path.parent / "run.json")
        wall = path.stat().st_mtime - datetime.fromisoformat(run["environment"]["time"]).timestamp()
        lines.append(f"| {name} | {len(rows)} | {sum(r['frames'] for r in rows)} | {sum(times):.2f} | "
                     f"{np.mean(times):.4f}/{np.percentile(times,95):.4f} | {wall:.2f} | {max(r['peak_gpu_mib'] for r in rows):.1f} |")
    lines.extend(["", "更新总时间包含前向、反向、同步和优化器更新；训练窗口由 run.json 的起始时间和完成文件时间戳计算，包含数据读取、周期验证和保存，排除模型/LPIPS 初始化。"
                  "分阶段恢复时窗口只对应最后一次恢复；新增 timing_*.json 另存每次完整训练进程耗时，包含初始化。所有逐步原始耗时和显存均保留。"])


def benchmark_table(lines):
    lines.extend(["", "| 模型/步 | K | Encoder/Sender/Receiver FPS | 联合 FPS | 联合 P50/P95 ms | GPU MiB | 参数量 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for path in sorted(ROOT.glob("S*/benchmark_*.json")):
        result = read(path)
        for k, row in result["budgets"].items():
            t = row["timing"]
            lines.append(f"| {path.parent.name}/{result['completed_steps']} | {k} | {t['encoder']['fps']:.2f}/{t['sender']['fps']:.2f}/{t['receiver']['fps']:.2f} | "
                         f"{t['joint']['fps']:.2f} | {t['joint']['p50_ms']:.3f}/{t['joint']['p95_ms']:.3f} | {row['peak_gpu_mib']:.1f} | {result['parameters']} |")
        lines.extend(["", "| K | 帧段 | 帧数 | Encoder/Sender/Receiver FPS | 联合 FPS | 联合 P50/P95 ms | bpp |",
                      "|---:|---|---:|---:|---:|---:|---:|"])
        for k, row in result["budgets"].items():
            for group, segment in row["groups"].items():
                t = segment["timing"]
                lines.append(f"| {k} | {group} | {segment['frames']} | {t['encoder']['fps']:.2f}/{t['sender']['fps']:.2f}/{t['receiver']['fps']:.2f} | "
                             f"{t['joint']['fps']:.2f} | {t['joint']['p50_ms']:.3f}/{t['joint']['p95_ms']:.3f} | {segment['actual_bpp']:.8f} |")
        fusion = result["fusion"]
        if fusion["enabled"]:
            lines.extend(["", f"{path.parent.name} 的 Cross-Scale Fusion 单独实测：{fusion['parameters']} 参数，"
                          f"{fusion['conv_macs']/1e6:.3f} M conv MACs，P50/P95 {fusion['latency']['p50_ms']:.4f}/{fusion['latency']['p95_ms']:.4f} ms。"
                          "卷积 FLOPs 计数不含上采样、归一化、激活及加法；模块计时包含全部操作。", ""])
    lines.extend(["", "吞吐在无其他 GPU 计算进程时测量，使用同一连续 64 帧和独立模型收发端；Sender 包含本地解码。联合吞吐是顺序 Encoder + 打包/本地解码 + Receiver；"
                  "排除磁盘读取、H2D、指标和校验。每帧耗时及 I/P 类型保存在 benchmark JSON，可重算分段 P50/P95。"])


def report():
    decision = read(ROOT / "decision_A.json")
    calibration = read(ROOT / "calibration.json")
    values = measurements()
    lines = ["# AdapTok-HR Unified RGB–Perceptual 实验报告", "",
             f"实验 A 选中 **{decision['winner']}**。双种子平均原生 LPIPS 均改善：**{decision['unified_improves_both_seeds']}**；逐预算质量保护门槛通过：**{decision['preferred_training_gate_pass']}**。",
             "实验 A 仅证明 D3 父模型上的 300 次统一目标微调结果，不能外推为从头训练结论。所有数字来自本分支实际执行。", "",
             "## 架构与目标", "",
             "原生 1920×1080 整帧输入，底部反射填充至 1088，输出裁回 1080。保留轻量卷积、S32/S64 局部窗口注意力、共享 8192×16 VQ、"
             "Base 135 + Medium 510 + Fine 2040 候选位置顺序、K=256/512/1024/2048、GOP32 和 13-bit 固定结构码流。模型只从离散 ID 对应的码向量重建；STE 仅用于反向传播。", "",
             "U0 保留原有 RGB L1+MSE、全分辨率 Alex LPIPS、VQ、temporal、monotonic、suffix_detail 和 high-frequency detail 损失，"
             "使用与 U1/U2 相同的新预算计划以隔离目标效应。U1/U2/U3 统一为 L1_RGB + λp×LPIPS + budget-aware VQ；没有层级颜色/亮度标签、专用后缀目标、单调正则或 GAN。", "",
             f"训练 LPIPS 是 480×270 全局视图和同位置原生 256×256 视图的等权均值。Alex λ={calibration['chosen_weights']['alex']}；"
             f"VGG λ={calibration['chosen_weights']['vgg']}。VGG 权重由 Train-only 冻结父模型 RGB 感知梯度范数校准；这不意味着两个网络处于相同损失尺度。", "",
             "每帧单次编码。独立预算 RNG 每帧固定消费配对概率、相邻边和方向三个随机数；所有组主预算完全匹配。"
             "标准组 50% 双预算，U3 为 75%；配对从三个相邻预算边均匀抽取。为让 U3 仅增加辅助解码，单预算帧也沿用同一主预算提案，端点主预算概率 1/6、中间预算 1/3。"
             "两个解码平均重建及 VQ 损失，只有主解码推进真实参考。裁剪使用独立 RNG，坐标只抽一次、两个预算共用，checkpoint 严格保存并恢复。", "",
             "| B 组 | local_residual | Encoder 输出后的融合 | Token 投影 |", "|---|---|---|---|",
             "| N0 | 开 | 无 | 原逐尺度 1×1→16D |", "| N1 | 开 | Cross-Scale | 融合后逐尺度 1×1→16D |", "| N2 | 关 | Cross-Scale | 融合后逐尺度 1×1→16D |", "",
             "Cross-Scale 在 S128→S64→S32 做 1×1 lateral/bridge、逐级双线性上采样相加、Depthwise Residual Block；不改变位置表或码流，不加入全帧自注意力。"
             "默认关闭且不写入旧模型配置序列化，因此旧模型指纹不变。新模块初始化不消费公共权重的 RNG，N0/N1/N2 的公共初始张量逐项匹配。", "",
             "所有组使用 AdamW，lr=1e-4、weight decay=0.01、gradient clip=1、BF16、TBPTT=2、batch_sequences=1；"
             "A 从各自 1300 步父检查点继承全部优化器状态。B 从头初始化，两独立种子；300/1000 步前四帧 clip，后续获准阶段采用 16/32 帧。", "",
             "种子计划：S1 模型/数据/预算/裁剪 = 20261118/20261119/20261120/20261201；S2 = 20261128/20261129/20261130/20261211。全部实际配置与继承检查点哈希逐组保存。", "",
             "## 正式验证协议", "",
             "全部质量指标来自完整 1080p float32 的 LPIPS-Alex v0.1 / DISTS 和真实离散码流。固定 Validation：blackswan 50 帧、car-roundabout 64 帧、dog 60 帧，共 174 帧。"
             "每个预算有独立 64 帧 car-roundabout 闭环，短源只使用其实际长度；没有重复帧补足。Cb/Cr 使用 BT.709 full-range centered 色差分量；方差保留率为逐帧空间色差方差总和之比。"
             "I/P 分开报告，late_GOP 指每个 GOP 的第 24–31 帧；原始记录另有第 48 帧后的序列晚期结果。", "",
             "同状态比较重放真实 K2048 历史后截断实际 ID 前缀。整个 Fine [645:2048] 的置乱和跨视频替换保留 Base/Medium，"
             "增量后缀 [1024:2048] 另存原有控制。跨视频只在三个源共同存在的前 50 帧比较，全部干预不推进参考。", "",
             "## 两种子与各预算实测"]
    quality_table(lines, values)
    lines.extend(["", "## 状态一致性与 Fine 信息"])
    fine_table(lines, values)
    lines.extend(["", "## I / P / GOP 晚期"])
    groups_table(lines, values)
    lines.extend(["", "## 训练时间与显存"])
    resources_table(lines)
    lines.extend(["", "## 独立吞吐与融合开销"])
    benchmark_table(lines)
    for path in sorted(ROOT.glob("decision_B_*.json")):
        result = read(path)
        lines.extend(["", f"## 从头训练 {path.stem.rsplit('_',1)[1]} 步门槛", "",
                      f"继续下一阶段：**{result['advance']}**。完整逐预算及稳定性检查见 `{path.name}`。"])
        if not result["advance"]:
            lines.append("保留当前负结果，停止下一阶段扩展；不以更长训练替代结构诊断。")
    lines.extend(["", "## 图、原生重建与可复现性", "",
                  "`figures/` 的 PDF/SVG/PNG 和 source_data.json 使用实际 bpp 与完整原生验证指标；两种子单独绘制，不将相关视频帧当成独立重复。"
                  "每个验证目录 `samples/` 保存未经增强的原生 1920×1080 PNG，包含 I 帧及 P 帧（1、31、63）。这些图使用同一真实 K2048 历史，显示各离散 ID 前缀在相同状态下的重建；低预算 P 图不代表其独立预算闭环。正式率失真表使用独立预算闭环。PNG 为 8-bit 显示，指标使用原始 float32 重建。",
                  "", "全部配置、命令和门槛见 README.md、protocol.json；raw_records.tar.gz 保留逐帧指标、实际 packets、训练日志、测试、checkpoint 元数据及 SHA256。"
                  "完整权重和优化器状态保存在独立 checkpoints/adaptok_hr_unified。旧 Fine Detail v2 与 Perceptual Recovery 检出目录没有修改。"])
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    (PACKAGE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    report()
