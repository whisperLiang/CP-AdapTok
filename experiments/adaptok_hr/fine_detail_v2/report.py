"""Generate the delivery report from the measured artifacts and their preregistered gates."""


from .run import PACKAGE, PROTOCOL, ROOT
from .study import read


def summaries():
    for p in sorted(ROOT.glob("*/validation_*_*/summary.json")):
        yield p.parents[1].name, read(p)


def percent(value):
    return f"{100 * value:.2f}%"


def decoder_contrasts(decision):
    winner = decision["decoder_winner"]
    lines = ["", f"### 选中方案 {winner} 的 K2048 绝对画质差值", "",
             "| 种子 | 对照 | ΔPSNR (dB) | ΔLPIPS | ΔDISTS |", "|---|---|---:|---:|---:|"]
    for seed, arms in decision["seeds"].items():
        for key, label in (("control", "同步数 C0"), ("old", "历史 HR")):
            delta = arms[winner]["quality"]["absolute_delta"][key]
            lines.append(f"| {seed} | {label} | {delta['psnr_global']:+.5f} | {delta['lpips']:+.6f} | {delta['dists']:+.6f} |")
    lines.extend(["", "### 同种子各项 Decoder 改动相对 D0 的代价和收益", "",
             "每行比较同一 1300 步、K2048、同一验证帧。延迟差取两组独立 64 帧实测的平均每帧差值。"
             "正的延迟差表示增加耗时。", "",
             "| 种子 | 改动 | ΔPSNR (dB) | ΔLPIPS | ΔDISTS | Δ边缘 MSE | Δ高频 L1 | Δ接收 ms | Δ联合 ms | Δ参数量 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for seed, arms in decision["seeds"].items():
        root = ROOT / f"S{seed}_D0"
        base = read(root / "validation_1300_64/summary.json")["closed_loop"]["2048"]
        timing = read(root / "benchmark_1300_1.json")
        for arm in arms:
            if arm == "D0":
                continue
            root = ROOT / f"S{seed}_{arm}"
            current = read(root / "validation_1300_64/summary.json")["closed_loop"]["2048"]
            bench = read(root / "benchmark_1300_1.json")
            quality = [current[k] - base[k] for k in ("psnr_global", "lpips", "dists", "edge_mse", "high_frequency_l1")]
            latency = [bench["budgets"]["2048"]["timing"][k]["mean_ms"] - timing["budgets"]["2048"]["timing"][k]["mean_ms"]
                       for k in ("receiver", "joint")]
            lines.append("| " + " | ".join([seed, arm, *(f"{v:+.6f}" for v in quality),
                         *(f"{v:+.3f}" for v in latency), f"{bench['parameters'] - timing['parameters']:+d}"]) + " |")
    lines.extend(["", "### 原图平坦区域的额外高频误差", "",
                  "在完整 174 帧的 K2048 独立闭环包上重新解码，逐帧重建 MSE 与已封存值一致。"
                  "原图 3×3 邻域各 RGB 通道的极差均不超过 2 个 8-bit 灰阶时，记为平坦像素。"
                  "以反射边界 3×3 均值滤波测量重建误差的高频分量，并统计任一通道误差超过 4 个灰阶的比例。"
                  "这项描述性指标用于检查平坦区域的噪声和伪纹理，选模沿用预登记规则。", "",
                  "| 种子 | 配置 | 原图平坦像素占比 | 平坦区高频 MAE (RGB 0–1) | 超过 4 灰阶的平坦像素 |",
                  "|---|---|---:|---:|---:|"])
    for seed, arms in decision["seeds"].items():
        for arm in ["C0", *arms]:
            pooled = read(ROOT / f"S{seed}_{arm}" / "flat_regions_1300.json")["pooled"]
            lines.append(f"| {seed} | {arm} | {percent(pooled['flat_fraction'])} | {pooled['flat_hp_mae']:.7f} | "
                         f"{100 * pooled['over_4_levels_fraction']:.3f}% |")
    return lines


def report():
    decision = read(ROOT / "decision.json")
    winner = decision["winner"]
    verdict = "两种子均确认后缀携带画面内容。" if decision["information_both_seeds"] else "已触发预登记的 1000 步信息停止条件。"
    lines = ["# AdapTok-HR 新增 Fine 后缀实验报告", "",
             f"300 步筛选选中了 **{winner}**。{verdict}", "",
             "R00 为 A1 对照，R10 加入局部空间残差，R01 加入训练期后缀监督，R11 同时启用两项。"
             "主实验基于 ab0b85b，固定共享球面码本、逐预算 VQ、四帧展开和 TBPTT2。"
             "K1024 已有 379 个 Fine 位置，干预仅作用于随后新增的 1024 个位置。", "",
             "## 实际码流与内容干预", "",
             "每个预算有独立的收发模型与闭环。另用同一 K2048 历史重放四档真实前缀。"
             "打乱后缀测位置贡献，同位置跨视频替换测内容相关性，所有干预沿用原参考。"
             "跨来源仅比较三源共有帧号。短验证为 24 帧，长验证为 174 帧，其中同位置跨视频替换使用 150 帧。", "",
             "| 配置 | 步数 | 验证帧 | 后缀码数 / perplexity / 最大占比 | 置乱 ΔMSE | 跨视频 ΔMSE | 同向来源 | 同状态 ΔdB | 闭环 ΔdB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    entries = sorted(summaries(), key=lambda pair: (pair[1]["steps"], pair[0]))
    confirmed = [r for name, r in entries if name.endswith("_" + winner) and r["steps"] == 1000]
    if confirmed:
        same = " / ".join(f"{r['same_state_gain_db']:+.3f}" for r in confirmed)
        loop = " / ".join(f"{r['closed_loop_gain_db']:+.3f}" for r in confirmed)
        lines[3:3] = [f"1000 步时 K2048 相对 K1024 的同状态增益依种子顺序为 **{same} dB**，"
                      f"独立闭环为 **{loop} dB**。每个 K2048 码流包均为 3470 字节。", ""]
    controls = [r for name, r in entries if name.endswith("_R00") and r["steps"] == 1000]
    if len(controls) == 2:
        first, second = controls
        if not first["information"]["pass"] and second["information"]["pass"]:
            net = " / ".join(f"{r['same_state_gain_db'] - c['same_state_gain_db']:+.3f}"
                             for r, c in zip(confirmed, controls))
            lines[5:5] = [f"原结构对照的第一个种子使用 {first['suffix']['used_codes']} 个后缀码字。"
                          f"第二个种子使用 {second['suffix']['used_codes']} 个，也通过两类内容干预，"
                          f"同状态增益为 {second['same_state_gain_db']:.3f} dB。"
                          f"{winner} 相对各自对照额外提高同状态预算增益 {net} dB。"
                          "初始模型、数据与预算种子一起变化，本实验无法进一步区分三者的贡献。", ""]
    if (ROOT / "decoder_decision.json").exists():
        d = read(ROOT / "decoder_decision.json")
        adoption = "通过双种子画质验收" if d["adoption_ready"] else "未同时通过双种子画质验收"
        lines[5:5] = [f"Decoder 阶段选择 **{d['decoder_winner']}**，该配置{adoption}。"
                      "逐种子、逐档质量差值以及独立吞吐见后文。", ""]
    for name, row in entries:
        s, info = row["suffix"], row["information"]
        p = info["pooled"]
        lines.append(f"| {name} | {row['steps']} | {p['frames']} | {s['used_codes']} / {s['perplexity']:.2f} / {percent(s['maximum_code_fraction'])} "
                     f"| {p['shuffle']:.8f} | {p['cross']:.8f} | {info['agreeing_sources']}/3 | {row['same_state_gain_db']:+.5f} | {row['closed_loop_gain_db']:+.5f} |")
    pilot = {name.split("_")[1]: row for name, row in entries if row["steps"] == 300}
    lines.extend(["", "### 同步数两因素对照", "",
                  "下表在另一因素固定时单独切换一项，均为首种子 300 步的配对差值。", "",
                  "| 新组 − 对照 | 同状态预算增益变化 (dB) | K2048 闭环 ΔPSNR | ΔLPIPS | ΔDISTS |",
                  "|---|---:|---:|---:|---:|"])
    for a, b in (("R10", "R00"), ("R11", "R01"), ("R01", "R00"), ("R11", "R10")):
        ra, rb = pilot[a], pilot[b]
        qa, qb = ra["closed_loop"]["2048"], rb["closed_loop"]["2048"]
        lines.append(f"| {a} − {b} | {ra['same_state_gain_db'] - rb['same_state_gain_db']:+.5f} | "
                     f"{qa['psnr_global'] - qb['psnr_global']:+.5f} | {qa['lpips'] - qb['lpips']:+.6f} | {qa['dists'] - qb['dists']:+.6f} |")
    lines.extend(["", "在本轮同步数的四组筛选中，保留原投影的 R00、R01 后缀均为单码。"
                  "扣除 S32 投影的局部均值后，R10 和 R11 的真实后缀均通过了内容干预。"
                  "因此局部空间分量在这个种子和训练长度下足以打破单码状态，尚不能把它推广为所有种子的唯一成因。"
                  "前一轮已有冻结权重、连续旁路及预算感知和分层码本的独立对照。"
                  " [第一轮表征诊断与负结果](../../../docs/adaptok_hr_fine_recovery_report.md)"])
    lines.extend(["", "## 多预算独立闭环画质", "",
                  "| 配置 | 步数 | K | 全局 PSNR (dB) | LPIPS | DISTS | 边缘 MSE | 高频 L1 | 实际 bpp |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    historical = read(PROTOCOL["historical_validation"])
    for name, row in [("历史 HR", {"steps": 1000, "closed_loop": historical["closed_loop"]}), *entries]:
        for k, q in row["closed_loop"].items():
            edge = f"{q['edge_mse']:.6f}" if "edge_mse" in q else "—"
            hp = f"{q['high_frequency_l1']:.6f}" if "high_frequency_l1" in q else "—"
            lines.append(f"| {name} | {row['steps']} | {k} | {q['psnr_global']:.5f} | {q['lpips']:.6f} | {q['dists']:.6f} | {edge} | {hp} | {q['actual_bpp']:.8f} |")
    lines.extend(["", "## 双种子与采用条件", "",
                  "按每个种子的同步数对照逐档保护 K256、K512、K1024，所有预算按相同输入帧 SHA256 配对。"
                  "晚期取每源从帧 48 开始的相同帧。PSNR 容差 0.05 dB，LPIPS 与 DISTS 容差各 0.005。", "",
                  "| 种子 | 内容 | 同状态 ≥0.10 dB | 闭环增益 | 感知随预算改善 | 绝对三指标 | 低预算及晚期 | 包长 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    word = {True: "通过", False: "未过"}
    for seed, r in decision["seeds"].items():
        q = r["quality"]["checks"]
        lines.append("| " + " | ".join([seed, word[r["information"]["pass"]], *(word[q[k]] for k in
                         ("same_state_gain", "closed_loop_gain", "perception_with_budget", "absolute_quality", "protected", "rate"))]) + " |")
    lines.extend(["", "### 低预算及晚期相对同种子对照的差值", "",
                  "| 种子 | 预算 / 帧段 | ΔPSNR | ΔLPIPS | ΔDISTS |", "|---|---|---:|---:|---:|"])
    for seed, r in decision["seeds"].items():
        for key, v in r["quality"]["protected_delta"].items():
            lines.append(f"| {seed} | {key} | {v['psnr_global']:+.5f} | {v['lpips']:+.6f} | {v['dists']:+.6f} |")
    lines.extend(["", "## 冻结 Train 首帧的早期表征", "",
                  "同一 bear I 帧、K2048，从真实离散前向取梯度范数。后缀监督梯度包含配置权重。", "",
                  "| 配置 | 步 | 新增位置码数 | 位置平均余弦 | 重建梯度 | 辅助梯度 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for p in sorted(ROOT.glob("*/probes/*.json")):
        r = read(p)
        v = r["levels"]["incremental_suffix"]
        lines.append(f"| {p.parents[1].name} | {r['step']} | {v['used_codes']} | {v['vq_input']['mean_position_cosine']:.6f} | "
                     f"{v['reconstruction_gradient_norm']:.7g} | {v['weighted_suffix_gradient_norm']:.7g} |")
    if (ROOT / "decoder_report.md").is_file():
        lines.extend(["", (ROOT / "decoder_report.md").read_text()])
        lines.extend(decoder_contrasts(read(ROOT / "decoder_decision.json")))
        lines.extend(["", "### Decoder 起点的原生 BF16 复核", "",
                      "第一版残差掩码将参考特征提升为 FP32，零初始化时第二帧已出现像素差异。"
                      "保留了该轮的两个完整训练和一个中断训练，其结果排除在选模和所有表格之外。"
                      "修正后从相同的第 1000 步父检查点重跑全部续训。每个分叉在首个新增更新前"
                      "均以真实包在四档预算验证连续 I/P 帧，传输 ID、RGB 和 BF16 参考特征逐位一致。"
                      " [首次失败原始记录](evidence/superseded_bf16_reference.json)"])
    elif not decision["decoder_stage_allowed"]:
        lines.extend(["", "信息验收触发停止条件，Decoder 支路、扩训和 30 FPS 优化止于该条件。"])
    if (ROOT / "visual_review.md").is_file():
        lines.extend(["", (ROOT / "visual_review.md").read_text()])
    if (ROOT / "performance_report.md").is_file():
        lines.extend(["", (ROOT / "performance_report.md").read_text()])
    acceptance = read(ROOT / "acceptance.json")
    data = read(ROOT / "S1_R00/run.json")
    lines.extend(["", "## 复现与工程验收", "",
                  f"实机测试 {acceptance['tests']['tests']} 项，通过断言捕获 {acceptance['mutations_killed']} 个变异。"
                  f"复算 {len(acceptance['checkpoints'])} 个优化器检查点 SHA256，重读 "
                  f"{sum(v['packets_reparsed'] for v in acceptance['validations'].values())} 个验证包并核对前缀、输入身份和因果链。",
                  "图表按种子和步数分别绘制。相邻视频帧有关联，报告给出原始逐帧值，统计单位保留到来源。"
                  "全部验证限预先固定的三个 DAVIS 来源。封存 Test 未参与本轮选型。", "",
                  f"训练数据清单身份为 `{entries[0][1]['identity']['manifest']}`。"
                  f"从头实验的生产代码 SHA256 为 `{data['environment']['source_sha256']}`。"
                  "每个实际检查点的权重、模型和码本指纹随证据归档。", "",
                  "[运行命令](README.md) · [工作日志](work_log.md) · [验收明细](evidence/acceptance.json) · [原始证据索引](evidence/archives.json)", "",
                  "## 实测曲线与原生重建", ""])
    featured = {"S1_1000_rd_psnr_global", "S2_1000_rd_lpips", "S1_1300_rd_psnr_global", "S2_1300_rd_lpips"}
    plots = sorted((ROOT / "figures").glob("*.png"))
    for p in (p for p in plots if p.stem in featured):
        lines.extend([f"![{p.stem}](evidence/figures/{p.name})", ""])
    lines.extend(["[全部曲线与单图 PDF、SVG](evidence/figures) · [对应的实测坐标](evidence/figures/plotted_values.json)", ""])
    for p in (p for p in plots if p.stem.endswith("_reconstruction")):
        lines.extend([f"![{p.stem}，上排全帧缩略图，下排相同位置原始像素裁剪](evidence/figures/{p.name})", ""])
    for p in sorted((PACKAGE / "evidence/samples").glob("*/car-roundabout_0000_K2048.png")):
        lines.append(f"- [{p.parent.name} 原生 1080p 四预算图]({p.parent.relative_to(PACKAGE)})")
    (PACKAGE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    report()
