"""Generate the delivery report from the measured artifacts and their preregistered gates."""


from .run import PACKAGE, PROTOCOL, ROOT
from .study import read


def summaries():
    for p in sorted(ROOT.glob("*/validation_*_*/summary.json")):
        yield p.parents[1].name, read(p)


def percent(value):
    return f"{100 * value:.2f}%"


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
    elif not decision["decoder_stage_allowed"]:
        lines.extend(["", "信息验收触发停止条件，Decoder 支路、扩训和 30 FPS 优化止于该条件。"])
    if (ROOT / "visual_review.md").is_file():
        lines.extend(["", (ROOT / "visual_review.md").read_text()])
    if (ROOT / "performance_report.md").is_file():
        lines.extend(["", (ROOT / "performance_report.md").read_text()])
    acceptance = read(ROOT / "acceptance.json")
    lines.extend(["", "## 复现与工程验收", "",
                  f"实机测试 {acceptance['tests']['tests']} 项，通过断言捕获 {acceptance['mutations_killed']} 个变异。"
                  f"复算 {len(acceptance['checkpoints'])} 个优化器检查点 SHA256，重读 "
                  f"{sum(v['packets_reparsed'] for v in acceptance['validations'].values())} 个验证包并核对前缀、输入身份和因果链。",
                  "图表按种子和步数分别绘制。相邻视频帧有关联，报告给出原始逐帧值，统计单位保留到来源。"
                  "全部验证限预先固定的三个 DAVIS 来源。", "",
                  "[运行命令](README.md) · [工作日志](work_log.md) · [验收明细](evidence/acceptance.json) · [原始证据索引](evidence/archives.json)", "",
                  "## 实测曲线与原生重建", ""])
    for p in sorted((ROOT / "figures").glob("*.png")):
        lines.extend([f"![{p.stem}](evidence/figures/{p.name})", ""])
    for p in sorted((PACKAGE / "evidence/samples").glob("*/car-roundabout_0000_K2048.png")):
        lines.append(f"- [{p.parent.name} 原生 1080p 四预算图]({p.parent.relative_to(PACKAGE)})")
    (PACKAGE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    report()
