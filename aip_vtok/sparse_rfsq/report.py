"""Generate Markdown tables and figures from per-frame and per-update evidence."""

from collections import defaultdict
from statistics import median

from aip_vtok.c1.scan import BUDGETS
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash

from .evaluation import CLOSED, FIXED, S1, S2, summaries
from .figures import gradient_cosines, s1_curves, s2_gains
from .protocol import (
    CALIBRATION,
    LIMITS,
    OUTPUT,
    PILOTS,
    PROTOCOL,
    REPORT,
    S1_GATE,
    S2_GATE,
    SCAN_LOCK,
    TEST_GATE,
    WORK_LOG,
)
from .receiver import B2
from .scan import LEGACY

DATA = "../results/aip_vtok/sparse_rfsq"
SCOPE = {"validation": "Validation 174 帧", "horizon": "car-roundabout 64 帧"}
METRICS = ("psnr", "lpips", "dists")
NORMS = ("mse", "lpips", "anchor")
PAIRS = ("mse/lpips", "mse/anchor", "lpips/anchor")


def table(headers, rows):
    def line(items):
        return "| " + " | ".join(map(str, items)) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def arm(row):
    label = f"{row['strategy']} {row['method']}"
    return label if row["variant"] == "base" else f"{label} {row['variant']}"


def metrics_row(point):
    s = point["all"]
    return (arm(point), point["history"], point["budget"], s["frames"],
            f"{s['level1_symbols']:.0f}/{s['level2_symbols']:.0f}", f"{s['mse']:.7f}",
            f"{s['psnr_global_mse']:.3f}", f"{s['psnr_frame_mean']:.3f}",
            f"{point['I']['psnr_global_mse']:.3f}", f"{point['P']['psnr_global_mse']:.3f}",
            f"{s['lpips']:.5f}", f"{s['dists']:.5f}", f"{s['cb_mse']:.3e}/{s['cr_mse']:.3e}",
            f"{s['cb_var_retention_pct']:.2f}/{s['cr_var_retention_pct']:.2f}",
            f"{s['cb_mean_abs_bias']:.3e}/{s['cr_mean_abs_bias']:.3e}")


def metrics_table(points):
    headers = ("方法", "历史", "K", "帧数", "L1/L2¹", "RGB MSE", "全局 PSNR", "逐帧 PSNR 均值", "I PSNR",
               "P PSNR", "LPIPS-Alex", "DISTS", "Cb/Cr MSE", "Cb/Cr 方差保留 %", "Cb/Cr 平均绝对偏差")
    return table(headers, (metrics_row(p) for p in points))


def efficiency_table(entries):
    return table(("方法", "历史", "K", "RGB MSE 下降", "增加 16-bit 负载 bit", "增加经验熵估算 bit",
                  "ΔMSE/Δbit 16-bit", "ΔMSE/Δbit 经验熵"),
                 ((arm(r), r["history"], f"{r['lower']}→{r['upper']}", f"{r['mse_drop']:+.5e}",
                   f"{r['fixed_additional_bits']:.0f}", f"{r['entropy_additional_bits']:.2f}",
                   f"{r['delta_mse_per_fixed_bit']:+.5e}", f"{r['delta_mse_per_entropy_bit']:+.5e}") for r in entries))


def backslide_table(entries):
    grouped = defaultdict(dict)
    for r in entries:
        key = (arm(r), r["history"], f"{r['lower']}→{r['upper']}")
        grouped[key][r["metric"]] = f"{r['violations']}/{r['pairs']} · {r['fraction']:.2%} · {r['max_backslide']:.5f}"
    return table(("方法", "历史", "K", "PSNR (dB)", "LPIPS", "DISTS"),
                 ((*key, *(value[metric] for metric in METRICS)) for key, value in grouped.items()))


def horizon_tables(summary):
    refs = summary["reference_frames"]
    ends = table(("方法", "历史", "帧号", "参考 K", "参考潜在 RMS", "投影特征 RMS", "收发同步", "状态 SHA256 前 12 位"),
                 ((arm(r), r["history"], r["frame"], r["reference_budget"], f"{r['reference_latent_rms']:.6f}",
                   f"{r['projected_feature_rms']:.6f}", "通过" if r["state_sync"] else "失败",
                   r["rx_reference_sha256"][:12]) for r in refs))
    late = table(("方法", "历史", "K", "全局 PSNR", "逐帧 PSNR 均值", "LPIPS", "DISTS", "参考 RMS 均值"),
                 ((arm(r), r["history"], r["budget"], f"{r['all']['psnr_global_mse']:.3f}",
                   f"{r['all']['psnr_frame_mean']:.3f}", f"{r['all']['lpips']:.5f}", f"{r['all']['dists']:.5f}",
                   f"{r['all']['reference_latent_rms']:.6f}") for r in summary["late"]))
    return ["### 第 31/63 帧参考状态", ends, "### 晚期第 48–63 帧", late]


def locked_selection(lock, data):
    pct = lock["sp3_refinement_pct"]
    rows = [(r["strategy"], r["refinement_pct"], r["sigma"], f"{r['score_db']:.4f}",
             "通过" if r["progressive"] else "失败") for r in data["selection"]["trials"]]
    return [f"Train 使用 {len(lock['holdout'])} 个独立视频，共 {len(lock['holdout']) * 4} 帧。"
            f"冻结 SP‑3 为 {100-pct}/{pct}（L1/L2），公共 σ 为 {lock['sigma']}，优胜扫描表为 **{lock['winner']}**。"
            f"锁定前 Validation 和 Test 图像读取数分别为 {lock['validation_images_before_lock']} 和 {lock['test_images_read']}。"
            f" [Train 原始 CSV]({DATA}/calibration.csv) · [扫描表]({DATA}/scan_lock.json)",
            table(("扫描表", "锚点后 L2 %", "σ", "I/P × K 平衡 PSNR (dB)", "渐进性"), rows)]


def s1_comparison(summary):
    by_point = {(r["strategy"], r["history"], r["budget"]): r["all"] for r in summary["validation"]["points"]
                if r["method"] == B2}
    rows = []
    for mode in (FIXED, CLOSED):
        for policy in ("SP-1", "SP-2", "SP-3"):
            curve = [by_point[policy, mode, k]["psnr_global_mse"] for k in BUDGETS]
            rows.append((mode, policy, *(f"{v:.3f}" for v in curve)))
    focus = table(("历史", "B2 扫描表", *BUDGETS), rows)
    gain = by_point["SP-1", CLOSED, 8192]["psnr_global_mse"] - by_point["SP-2", CLOSED, 8192]["psnr_global_mse"]
    last = {r["strategy"]: r for r in summary["validation"]["efficiency"]
            if r["method"] == B2 and r["history"] == CLOSED and r["upper"] == 8192}
    space, precision = (last[name]["delta_mse_per_fixed_bit"] for name in ("SP-1", "SP-2"))
    low_control = by_point[LEGACY, CLOSED, 510]["psnr_global_mse"]
    low_sp1 = by_point["SP-1", CLOSED, 510]["psnr_global_mse"]
    return ["### 空间覆盖与精修", focus,
            f"在 174 帧独立闭环的 K8192，SP‑1 相对 SP‑2 的全局 PSNR 差为 **{gain:+.3f} dB**。"
            "两表到 K4096 使用相同的一级槽位，最后 4096 个符号分别用于新空间观测和已到达槽位的残差。"
            f"该段每新增固定负载 bit 的 RGB MSE 下降量依次为 {space:+.5e} 和 {precision:+.5e}。",
            f"在同一独立闭环队列的 K510，封存 C1‑A1 为 {low_control:.3f} dB，SP‑1 为 {low_sp1:.3f} dB。"
            "S2 以 Train 优胜策略本身的 B2 作为学习恢复的对照。"]


def measurements(summary):
    parts = ["¹ L1/L2 按帧取均值后显示至最近整数。PSNR 按归一化 RGB 计算，I/P 列使用各自分项的全局 MSE。"
             " Cb/Cr 按 BT.709 分别计算。"]
    for scope, label in SCOPE.items():
        data = summary[scope]
        parts.extend([f"### {label}", metrics_table(data["points"]),
                      "回退单元格依次为次数/比较帧数、比例和最大单帧回退。", backslide_table(data["backslides"])])
    return [*parts, *horizon_tables(summary)]


def gate_table(decisions):
    keys = ("hard_integrity", "focus_psnr", "focus_lpips_dists", "late_quality", "reference_rms", "psnr_backslides")
    return table(("权重", "完整性", "重点 PSNR", "重点 LPIPS/DISTS", "晚期画质", "参考 RMS", "PSNR 回退", "总门控"),
                 ((label, *("通过" if checks[k] else "失败" for k in keys), "通过" if result["passed"] else "失败")
                  for label, result in decisions.items() for checks in (result["checks"],)))


def delta_table(decisions):
    return table(("权重", "队列", "历史", "K", "Δ全局 PSNR dB", "Δ逐帧 PSNR 均值 dB", "ΔLPIPS", "ΔDISTS"),
                 ((label, SCOPE.get(scope, "晚期 16 帧"), r["history"], r["budget"], f"{r['psnr_gain_db']:+.3f}",
                   f"{r['mean_frame_psnr_gain_db']:+.3f}", f"{r['lpips_change']:+.5f}", f"{r['dists_change']:+.5f}")
                  for label, trial in decisions.items() for scope, rows in trial["gains"].items() for r in rows))


def late_history_comparison(decisions):
    rows = []
    for variant, trial in decisions.items():
        modes = {(r["history"], r["budget"]): r for r in trial["gains"]["late"]}
        for k in BUDGETS:
            fixed, closed = modes[FIXED, k], modes[CLOSED, k]
            rows.append((variant, k, f"{fixed['psnr_gain_db']:+.3f}", f"{closed['psnr_gain_db']:+.3f}",
                         f"{closed['psnr_gain_db'] - fixed['psnr_gain_db']:+.3f}"))
    return ["### 闭环对晚期恢复增益的影响",
            "晚期第 48–63 帧按各自口径先计算 B3 相对 B2 的全局 PSNR 增益，再比较两种历史的增益。"
            "差值为负表示独立推进后晚期 PSNR 收益缩小。",
            table(("权重", "K", "固定参考增益 dB", "独立闭环增益 dB", "独立减固定 dB"), rows)]


def gradient_records(training):
    return {name: [{"step": row["step"], **row["gradient_diagnostics"]} for row in record["updates"]
                   if row["gradient_diagnostics"] is not None] for name, record in training.items()}


def cosine_cell(rows, pair):
    values = [r["cosines"][pair] for r in rows if r["cosines"][pair] is not None]
    return f"{median(values):+.3f} ({sum(x < 0 for x in values)}/{len(values)} 负)" if values else "未定义"


def gradient_tables(records):
    agg = table(("权重", "‖MSE‖", "‖LPIPS‖", "‖锚定‖", "MSE/LPIPS", "MSE/锚定", "LPIPS/锚定"),
                ((name, *(f"{median(r['norms'][k] for r in rows):.4e}" for k in NORMS),
                  *(cosine_cell(rows, k) for k in PAIRS)) for name, rows in records.items()))
    detail = table(("权重", "步数", "‖MSE‖", "‖LPIPS‖", "‖锚定‖", "MSE/LPIPS", "MSE/锚定", "LPIPS/锚定"),
                   ((name, r["step"], *(f"{r['norms'][key]:.4e}" for key in NORMS),
                     *(f"{r['cosines'][key]:+.4f}" if r["cosines"][key] is not None else "未定义" for key in PAIRS))
                    for name, rows in records.items() for r in rows))
    return ["### 共享主干梯度诊断", "每 50 步计算一个优化更新的 4 帧平均梯度，各损失项单独求导且保留原始权重前的范数。"
            "汇总使用 10 次观测的中位数。负余弦表示该更新的两项目标在共享主干中存在方向冲突。", agg, detail]


def diagnostic_interpretation(records):
    text = []
    for variant, rows in records.items():
        negative = sum((r["cosines"]["mse/lpips"] or 0) < 0 for r in rows)
        ratios = [PILOTS[variant] * r["norms"]["lpips"] / r["norms"]["mse"] for r in rows if r["norms"]["mse"]]
        text.append(f"{variant} 的 MSE–LPIPS 负余弦为 {negative}/{len(rows)}。"
                    f"加权 LPIPS 梯度范数相对 MSE 的中位比值为 {median(ratios):.3f}。")
    return " ".join(text) + " 这些观测用于定位多目标梯度冲突。空间恢复容量的影响需要独立的容量对照。"


def write_s1(lock):
    data = json_at(S1.with_suffix(".json"))
    gate = json_at(S1_GATE)
    if gate["s1_sha256"] != file_hash(S1.with_suffix(".json")):
        raise RuntimeError("S1 report evidence changed")
    summary = summaries(data["rows"])
    s1_curves(summary)
    parts = ["## S1 免训练扫描筛选", *locked_selection(lock, json_at(CALIBRATION.with_suffix(".json"))),
             f"S2 启动门控 **{'通过' if gate['released'] else '停止'}**。"
             f" {gate.get('reason') or 'Validation 复验确认 Train 优胜者的渐进性和参考 RMS。'}"
             f" [逐帧 CSV]({DATA}/s1.csv) · [门控]({DATA}/s1_gate.json) · "
             f"[独立闭环曲线]({DATA}/figures/s1_closed_rate_distortion.pdf)",
             *s1_comparison(summary), *measurements(summary), "### 相邻预算的 ΔMSE/Δbit",
             "每档从相同显示前缀计算固定 `16 × K` bit 与 Train 分组经验熵估算 bit。"
             "ΔMSE/Δbit 使用队列平均 RGB MSE 降幅除以队列平均新增负载。静态扫描表的逐帧侧信息为 0 字节。"
             "经验熵来自封存的 Train-only C1-Followup ID 直方图。账本中的帧头和级别包头是模拟开销，真实熵编码尚未实现。"]
    for scope, label in SCOPE.items():
        parts.extend([f"#### {label}", efficiency_table(summary[scope]["efficiency"])])
    return parts


def write_s2(lock):
    title = "## S2 500 步恢复 Pilot"
    if not S2_GATE.is_file():
        return [title, "S2 门控待生成。"]
    gate = json_at(S2_GATE)
    if not gate["started"]:
        return [title, f"S1 启动门控停止。S2 训练步数为 0。 [门控]({DATA}/s2_gate.json)"]
    data = json_at(S2.with_suffix(".json"))
    if gate["s2_sha256"] != file_hash(S2.with_suffix(".json")):
        raise RuntimeError("S2 report evidence changed")
    training = {name: json_at(OUTPUT / f"{name}_training.json") for name in PILOTS}
    diagnostics = gradient_records(training)
    decisions = gate["decisions"]
    s2_gains(decisions)
    gradient_cosines(diagnostics)
    qualified = "建议后续进入 S3" if gate["recommend_s3"] else "不建议进入 S3"
    conclusion = f"**{qualified}**。本轮执行至 S2，共完成 W1/W2 各 {training['W1']['steps']} 步。"
    scope = "优胜表 SP‑1 只发送 L1，二级到达掩码恒为零。S2 结果仅识别该空间扫描上的恢复收益。"
    detail = ("W1 与 W2 共享初始化哈希、500 个训练窗口及逐步预算和裁剪 RNG。每步监督 4 帧，使用各自 B3 参考。"
              "冻住的重建器通过 BF16 和非重入 checkpoint 回传输入梯度。同上下文参考损失使用 FP32 投影器。")
    return [title, conclusion, scope if lock["winner"] == "SP-1" else "", detail, s2_limits(), gate_table(decisions),
            f"[逐帧 CSV]({DATA}/s2.csv) · [S2 门控]({DATA}/s2_gate.json) · [闭环增益]({DATA}/figures/s2_closed_psnr_gain.pdf)",
            "表中 Δ 等于 B3 减同协议、同预算的 B2。LPIPS 和 DISTS 的负值表示改善。", delta_table(decisions),
            *late_history_comparison(decisions),
            *gradient_tables(diagnostics), diagnostic_interpretation(diagnostics),
            f"[W1 更新]({DATA}/W1_training.json) · [W2 更新]({DATA}/W2_training.json) · "
            f"[余弦轨迹]({DATA}/figures/s2_gradient_cosines.pdf)", *measurements(summaries(data["rows"]))]


def s2_limits():
    return ("事前门槛要求 Validation 的 K510/2048/8192 全局 PSNR 均为正增益，至少两档达到 +0.5 dB，"
            "三档 LPIPS 下降，DISTS 增量至多 +0.005。64 帧晚期五档的全局与逐帧均值 PSNR 下限均为 "
            f"{LIMITS['late_min_psnr_gain_db']:+.2f} dB，LPIPS/DISTS 增量上限为 +0.005。参考潜在 RMS 上限为 "
            f"{LIMITS['max_reference_latent_rms']:.1f}，晚期均值至多为同协议 B2 的 "
            f"{LIMITS['late_max_rms_relative_to_b2']:.2f} 倍。每个相邻档位的单帧 PSNR 回退上限为 1.0 dB "
            "和同协议 B2 最大回退加 0.25 dB 两者中的较大值。")


def command_table(records):
    return table(("任务", "命令", "退出码"),
                 ((r["stage"], f"`{r['command'] if isinstance(r['command'], str) else ' '.join(r['command'])}`",
                   r["exit_code"]) for r in records))


def work_log(protocol, tests):
    paths = sorted((OUTPUT / "audits").glob("*.json"))
    stages = sorted((json_at(p) for p in paths), key=lambda r: r["started"])
    log = ["# AIP‑VTok Sparse‑RFSQ 工作记录", f"基线 `{protocol['base_commit']}`。时间使用 Asia/Shanghai。",
           f"Train 像素读取前冻结协议和门槛于 {protocol['locked_at']}。 [协议]({DATA}/protocol.json)",
           table(("阶段", "开始", "结束", "退出码", "Train 帧", "Validation 帧", "Test 帧"),
                 ((r["stage"], r["started"], r["finished"], r["exit_code"], *(r.get("images_read", {}).get(s, 0)
                    for s in ("train", "validation", "test"))) for r in stages)),
           command_table(stages)]
    extra = OUTPUT / "commands.json"
    if extra.is_file():
        log.extend(["## 实现与验证命令", command_table(json_at(extra))])
    if tests:
        log.extend(["## 联合回归", command_table([{"stage": "pytest", **tests}]),
                    f"{tests['counts']['regression']} 项旧用例及 {tests['counts']['new']} 项新增用例。"
                    f" [JUnit]({DATA}/regression.xml) · [日志]({DATA}/regression.log)"])
    if (OUTPUT / "mutations.json").is_file():
        mutations = json_at(OUTPUT / "mutations.json")
        log.append(f"隔离变异检出 {sum(m['killed'] for m in mutations)}/{len(mutations)} 项。"
                   f" [变异记录]({DATA}/mutations.json) · [复杂度与重复审计]({DATA}/quality.json)")
    WORK_LOG.write_text("\n\n".join(log) + "\n")


def generate():
    protocol = json_at(PROTOCOL)
    lock = json_at(SCAN_LOCK)
    protected = json_at(OUTPUT / "protected.json")
    tests = json_at(TEST_GATE) if TEST_GATE.is_file() else None
    lines = ["# AIP‑VTok Sparse‑RFSQ",
             "`fixed_reference` 与 `fixed_reference_B3` 共用封存 C1 A1/B2 的 `I2048 / P510` 历史。"
             "`independent_closed` 和 `independent_closed_B3` 分别在各方法自己的发送端和接收端状态上运行同一逐帧参考预算。"
             "显示 K 与参考 K 独立。独立闭环若显示 K 小于参考 K，以 `max(K, Kref)` 记录维持参考所需负载。"
             "固定历史的共享参考负载另列。64 帧 `car-roundabout` 是 174 帧队列的一部分。",
             "扫描表以 Halton 顺序铺设每组的全图锚点。锚点后的一级组配额来自独立 Train 校准的梯度能量。"
             "二级槽位优先选择敏感度较高且已到达一级的组，组内沿用到达顺序。",
             f"Gate 0 投影器 SHA256 `{protected['controls']['projector']['sha256']}`。"
             f"保护清单覆盖 {protected['files']} 项，C1‑B 两颗种子均为 0 步，Test 图像读取数为 0。"
             f" [保护清单]({DATA}/protected.json) · [工作记录](aip_vtok_sparse_rfsq_work_log.md)"]
    if S1_GATE.is_file():
        lines.extend(write_s1(lock))
    else:
        lines.extend(["## S1 Train 校准", *locked_selection(lock, json_at(CALIBRATION.with_suffix(".json")))])
    lines.extend(write_s2(lock))
    if tests:
        lines.append(f"联合回归 {tests['counts']['total']} 项。 [JUnit]({DATA}/regression.xml)")
    REPORT.write_text("\n\n".join(line for line in lines if line) + "\n")
    work_log(protocol, tests)
