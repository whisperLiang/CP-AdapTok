"""Build the report and work log from recorded experiment and verification data."""

import json
import shlex
import statistics
from xml.etree import ElementTree

from aip_vtok.diagnostics.diag0.io import json_at

from .dense import ARMS
from .gates import HISTORIES, METHODS, deltas
from .io import OUTPUT, REPORT, WORK_LOG

RESULTS = "../results/aip_vtok/c1_followup/"
LABELS = {
    "A0": "A0 · Top-24 连续", "A1": "A1 · Top-24 原 FSQ", "A2": "A2 · Top-24 优化 FSQ",
    "A3": "A3 · Top-24 二级 Residual-FSQ", "A4-cont": "A4 · Top-36 连续", "A4-fsq": "A4 · Top-36 FSQ",
    "A5-cont": "A5 · Top-48 连续", "A5-fsq": "A5 · Top-48 FSQ",
}
HISTORY_NAMES = {"same_context": "同历史", "independent_closed": "独立闭环"}
MILESTONES = {"pilot": "seed=1、500 步", "seed1_final": "seed=1、1,000 步", "seed2_final": "seed=2、1,000 步"}


def raw(name):
    path = OUTPUT / f"{name}.json"
    return json_at(path) if path.is_file() else {}


def ref(name, filename):
    return f"[{name}]({RESULTS}{filename})"


def table(title, rows):
    return [
        "| " + " | ".join(title) + " |",
        "| " + " | ".join("---" for _ in title) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        "",
    ]


def f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def render_quality(summaries):
    return table(
        ("方案", "全局 PSNR", "逐帧均值 PSNR", "I PSNR", "P PSNR", "LPIPS-Alex", "DISTS", "潜在 MSE"),
        [(LABELS[arm], f(s["all"]["psnr_global_mse"]), f(s["all"]["psnr_frame_mean"]),
          f(s["I"]["psnr_global_mse"]), f(s["P"]["psnr_global_mse"]), f(s["all"]["lpips"], 5),
          f(s["all"]["dists"], 5), f(s["all"]["latent_mse"], 6)) for arm in ARMS for s in [summaries[arm]]],
    )


def render_chroma(summaries):
    columns = ("var_retention_pct", "mse", "mean_abs_bias")
    return table(
        ("方案", "Cb 方差 %", "Cb MSE", "Cb 均值绝对偏差", "Cr 方差 %", "Cr MSE", "Cr 均值绝对偏差"),
        [(LABELS[arm], *(f(summaries[arm]["all"][f"{plane}_{field}"], n)
                         for plane in ("cb", "cr") for field, n in zip(columns, (2, 7, 7)))) for arm in ARMS],
    )


def bit_row(name, row):
    return (
        LABELS[name], int(row["payload_bits_per_position"]), int(row["payload_bits"]),
        int(row["header_bytes"]), int(row["raw_total_bytes"]), f(row.get("entropy_payload_bits_per_position")),
        f(row.get("entropy_est_bits"), 0), f(row["encode_ms"], 2), f(row["symbol_encode_ms"], 2), f(row["decode_ms"], 2),
    )


def render_rates(summaries):
    return table(
        ("方案", "负载 bit/位置", "负载 bit/帧", "包头 B/帧", "定长 B/帧", "估算负载 bit/位置",
         "估算总 bit/帧", "编码 ms", "符号编码 ms", "解码 ms"),
        [bit_row(arm, summaries[arm]["all"]) for arm in ARMS],
    )


def histogram_rows(records):
    for record in records:
        dims = [dim for group in record["groups"] for dim in group["dimensions"]]
        total = sum(sum(d["counts"]) for d in dims)
        boundary = sum(d["counts"][0] + d["counts"][-1] for d in dims)
        saturated = sum(sum(d["counts"]) * d["tanh_saturation_ratio"] for d in dims)
        entropy = statistics.mean(group["id_entropy_bits"] for group in record["groups"])
        yield (LABELS[record["arm"]], record["frame_type"], record["level"], f(entropy),
               f(100 * boundary / total, 2), f(100 * saturated / total, 2))


def comparison_row(summaries, first, second, label):
    left = summaries[first]["all"]
    right = summaries[second]["all"]
    delta = deltas(left, right)
    return (label, f(delta["psnr_gain_db"]), f(delta["lpips_delta"], 5), f(delta["dists_delta"], 5),
            int(right["payload_bits_per_position"] - left["payload_bits_per_position"]),
            f(right["decode_ms"] - left["decode_ms"], 2))


def render_comparisons(summaries):
    pairs = [
        ("A1", "A2", "A2 − A1"), ("A3", "A5-fsq", "A5 FSQ − A3"),
        ("A4-cont", "A4-fsq", "A4 FSQ − A4 连续"), ("A5-cont", "A5-fsq", "A5 FSQ − A5 连续"),
        ("A1", "A3", "A3 − A1"), ("A1", "A4-fsq", "A4 FSQ − A1"), ("A1", "A5-fsq", "A5 FSQ − A1"),
    ]
    return table(("对照", "ΔPSNR dB", "ΔLPIPS", "ΔDISTS", "Δ负载 bit/位置", "Δ解码 ms"),
                 [comparison_row(summaries, first, second, label) for first, second, label in pairs])


def render_calibration(calibration, lock):
    if not calibration:
        return ["校准记录尚未生成。", ""]
    frames = calibration["frame_counts"]["train"]
    names = "、".join(calibration["shortlist"]) or "无"
    return [
        f"Train 校准包含 {len(calibration['plan'])} 个连续片段、{frames} 帧。"
        "按图像 SHA256 排除原 C1 拟合窗口和两个完整训练计划用过的帧，并跳过 Gate 0 校准视频。"
        "A2/A3 保持 A0/A1 的 24 个通道。Top-36/48 在这 24 个通道后按新 Train 梯度敏感度补足。",
        "",
        "原缩放为绝对值第 99.5 百分位除以 1.5。搜索覆盖尺度倍率、固定 μ-law 和逐组 levels 分配。"
        "每个 6D 符号保留 64,000 个可用 ID，并使用 16 bit 定长字。先按 Train 敏感度加权潜在误差筛选，"
        "再按 Train RGB MSE 选择每种 I/P 参数。A3 共用 A2 的一级量化器，二级输入是减去一级整数解码值的残差。",
        "",
        f"锁定候选为 **{names}**，锁定时间 {lock.get('locked_at', '—')}。"
        "64-bit 槽位要求相对 A1 至少 +0.3 dB，128-bit 槽位要求至少 +0.5 dB。"
        "两档均容许 LPIPS、DISTS 各最多增加 0.005。128-bit 排序使用绝对质量对应的相对 A1 差值，"
        "即 ΔPSNR − 10×ΔLPIPS − 10×ΔDISTS − 0.05×log₂(总编解码时长/A1)。"
        "Validation 只能否决冻结入选项。"
        f"{ref('搜索记录和分组参数', 'calibration.json')} · {ref('候选锁', 'candidates.json')}",
        "",
    ]


def render_phase1(validation, replay, admission):
    if not validation:
        return ["174 帧统一复验尚未生成。", ""]
    summaries = validation["summaries"]
    cohort = "、".join(f"{name.split(':')[-1]} {count} 帧" for name, count in validation["cohort"].items())
    retraced = replay["replay"]["comparisons"]
    total = summaries["A1"]["all"]["frames"]
    i, p = (summaries["A1"][kind]["frames"] for kind in ("I", "P"))
    passed = "、".join(admission.get("released", [])) or "无"
    return [
        f"{cohort}，合计 **{total} 帧**（I {i}、P {p}）。"
        "各组共享原 C1 A1 的量化参考历史，解码时对截断维度补零。"
        "分项 PSNR 使用该队列内各子集的全局 MSE，质量指标均来自同一队列。",
        "",
        f"A0、A1 与封存基线的最大逐帧 MSE 差分别为 "
        f"`{retraced['A0']['max_per_frame_mse_error']}` 和 `{retraced['A1']['max_per_frame_mse_error']}`。"
        f"{ref('历史回放原始记录', 'baseline.json')}",
        "",
        *render_quality(summaries), *render_chroma(summaries), *render_rates(summaries),
        "连续行按 FP32 负载计价。定长账本采用 16 B 帧头和 9 B 分级包头，每包至多 128 个符号。"
        "二级每个槽付两次符号成本，同槽一级 ID 是二级先决条件。通道表、尺度和 levels 配置为预共享协议。"
        "估算使用每帧、每组、每级 ID 边际熵，并在总 bit/帧加包头。当前数字属于诊断估算。"
        "计时为同步 CUDA 后的逐帧均值。编码含共同的 RGB 转换与骨干捕获，解码含整数解码和骨干 RGB 重建。",
        "",
        *render_comparisons(summaries),
        f"Phase 1 放行 **{passed}**。{ref('门控原始记录', 'phase1_gate.json')}",
        "",
        "下表按同一验证队列的 I/P 分组池化。每个维度的计数、熵、离散边界比例和真实 tanh 饱和率"
        f"（1 − tanh² < 0.05）在原始记录中逐级、逐组列出。{ref('逐帧 CSV', 'validation.csv')} · "
        f"{ref('完整直方图与逐帧 JSON', 'validation.json')}",
        "",
        *table(("方案", "帧型", "级", "ID 熵 bit/符号", "边界码 %", "tanh 饱和 %"),
               histogram_rows(validation["histograms"])),
    ]


def eval_table(record):
    rows = []
    for mode in HISTORIES:
        block = record["table"][mode]
        for method in METHODS:
            s = block[method]
            rows.append((HISTORY_NAMES[mode], method, f(s["psnr_global_mse"]), f(s["psnr_frame_mean"]),
                         f(s["lpips"], 5), f(s["dists"], 5), f(s["latent_mse"], 6)))
        gain = record["gate"]["histories"][mode]
        gap = deltas(block[METHODS[0]], block[METHODS[2]])
        failed = "、".join(check for check, passed in gain["checks"].items() if not passed) or "无"
        rows.append((HISTORY_NAMES[mode], "B1 − B0-quantized", f(gain["psnr_gain_db"]), "—",
                     f(gain["lpips_delta"], 5), f(gain["dists_delta"], 5), f(
                         block[METHODS[2]]["latent_mse"] - block[METHODS[1]]["latent_mse"], 6)))
        rows.append((HISTORY_NAMES[mode], "B1 − B0-cont", f(gap["psnr_gain_db"]), "—", f(gap["lpips_delta"], 5),
                     f(gap["dists_delta"], 5), f(block[METHODS[2]]["latent_mse"] - block[METHODS[0]]["latent_mse"], 6)))
        rows.append((HISTORY_NAMES[mode], f"门槛失败项 {failed}", "—", "—", "—", "—", "—"))
    return table(("历史", "方法", "全局 PSNR", "逐帧 PSNR 均值", "LPIPS-Alex", "DISTS", "潜在 MSE"), rows)


def render_actual_steps(training):
    return table(
        ("方案", "seed=1 步数", "seed=2 步数", "500 步", "seed=1 最终门槛", "seed=2 最终门槛"),
        [(arm, result["steps"]["seed1"], result["steps"]["seed2"],
          *(milestone_status(result[key]) for key in MILESTONES))
         for arm, result in training.get("candidates", {}).items()],
    )


def milestone_status(record):
    if record is None:
        return "未启动"
    return "通过" if record["gate"]["passed"] else "未通过"


def render_training(training):
    if not training:
        return ["受控训练尚无完整门控记录。", ""]
    lines = [
        "训练复用 C1-A 的原始 4 帧 detached-reference 因果展开、损失权重、AdamW、裁剪种子与逐步取样顺序。"
        "接收端模型输入来自整数 ID 解码。骨干与投影器冻结，训练逐帧检查发送、接收参考状态同步。",
        "",
        "500 步要求同历史 ≥+0.2 dB、独立闭环 ≥0 dB，两种历史的 LPIPS 均改善且 DISTS 增量各 ≤+0.005。"
        "1,000 步要求两种历史各 ≥+0.5 dB，并继续满足感知指标、逐帧同步和保护审计。"
        "两个种子通过各自的 1,000 步终验后才放行稀疏端。",
        "",
        *render_actual_steps(training),
    ]
    for arm, result in training["candidates"].items():
        for stage, label in MILESTONES.items():
            if record := result[stage]:
                path = f"dense_training/{arm}/seed{record['seed']}/step_{record['step']:04}"
                lines.extend([f"### {arm} · {label}", "", *eval_table(record),
                              f"{ref('逐帧 CSV', path + '.csv')} · {ref('指标、同步和 checkpoint SHA256', path + '.json')}", ""])
    if any(not result["pilot"]["gate"]["passed"] for result in training["candidates"].values()):
        lines += [
            "Pilot 的停止结论只适用于本轮 500 步设置。"
            "后续可在 Train 内锁定量化器与参考历史，逐项检查像素损失与感知损失对更新的影响，"
            "并分解未观测通道的误差和独立闭环误差，再决定是否调整训练目标。",
            "",
        ]
    return lines


def sparse_quality_rows(name, summaries):
    for point in summaries:
        for method, metric in point["metrics"].items():
            row = metric["all"]
            yield (name, point["budget_type"], point["budget"], method, f(row["unique_slots"], 1),
                   f(row["raw_bits"], 0), f(row["train_entropy_est_bits"], 0),
                   f(row["psnr_global_mse"]), f(row["lpips"], 5), f(row["dists"], 5))


def render_sparse(sparse):
    if not sparse:
        return ["C1-B-Followup 的最终稀疏门控记录尚未生成。", ""]
    if not sparse["started_validation"]:
        return [f"**停止于稠密门控。** C1-B-Followup 验证与训练均为 0，未生成新扫描表。"
                f"{ref('稀疏端门控记录', 'sparse_gate.json')}", ""]
    lines = ["C1-B-Followup 使用冻结的 Train 熵估算选择前缀，并逐帧复验三类预算下的稀疏率失真。", ""]
    for arm in sparse["released"]:
        name = f"c1b_followup/{arm}/validation"
        lines += table(("方案", "预算种类", "档", "方法", "槽数", "定长 bit/帧", "估算 bit/帧",
                        "PSNR", "LPIPS", "DISTS"), sparse_quality_rows(arm, raw(name)["summaries"]))
        lines += [ref("稀疏逐帧记录", name + ".json"), ""]
    return lines


def render_integrity(integrity, audits):
    lines = []
    if integrity:
        controls = integrity["controls"]
        lines += [
            f"`projector_w128` SHA256 为 `{controls['projector']['sha256']}`。"
            f"保护文件共 {integrity['files']} 项，汇总 SHA256 为 `{integrity['sha256']}`。",
            "",
            f"原 C1-B 两颗种子仍各为 0 步。实验进程合计读取封存 Test 图像 "
            f"**{sum(a.get('images_read', {}).get('test', 0) for _, a in audits)} 张**。"
            "所有成功阶段均核对了骨干与投影器的参数及缓冲区 SHA256。",
            "",
        ]
    result = test_result()
    if result is not None:
        total, passed, old, new = result
        lines += [f"联合回归共 {total} 项，{passed} 项通过（旧用例 {old} 项，Followup 用例 {new} 项）。"
                  f"{ref('JUnit', 'regression.xml')} · {ref('测试日志', 'regression.log')}", ""]
    if verified := raw("verification"):
        lines += [f"逐帧 CSV 和 JSON 核对了 {sum(verified['csv_json_paired_rows'].values())} 条记录。"
                  f"训练核对了 {sum(row['training_frame_states_verified'] for row in verified['phase2'].values())} 次"
                  f"发送与接收参考提交。{ref('最终核验', 'verification.json')}", ""]
    return lines


def test_result():
    path = OUTPUT / "regression.xml"
    if not path.is_file():
        return None
    cases = ElementTree.parse(path).findall(".//testcase")
    new = sum("test_c1_followup" in node.get("classname", "") for node in cases)
    passed = sum(all(node.find(tag) is None for tag in ("failure", "error", "skipped")) for node in cases)
    return len(cases), passed, len(cases) - new, new


def command_records():
    path = OUTPUT / "commands.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def work_log(protocol, lock, audits):
    lines = [
        "# AIP-VTok C1-Followup 工作记录", "",
        f"使用 Asia/Shanghai 时间。基线提交 `{protocol.get('base', 'a0f5eb8')}`，分支 "
        "`experiment/aip-vtok-c1-followup`。",
        "",
        "- 按 ub-mode 先写整数往返、重放、数据边界与门槛测试，记录初次失败后再实现。",
        "- 对齐历史训练计划，按图像哈希找出能用于候选拟合的未用 Train 片段。",
        f"- {protocol.get('locked_at', '—')} 锁定门槛。{lock.get('locked_at', '—')} 锁定 Train 参数和候选。",
        "- 基线逐帧匹配通过后对同一批 174 帧运行其他稠密候选。",
        "- 每个训练里程碑使用独立目录和原子检查点。下一段更新依赖前一门槛，稀疏入口再次验证双种子记录。",
        "",
        *table(("阶段", "开始", "结束", "实验阶段退出码", "Train 读取帧数", "Validation 读取帧数", "Test 读取帧数"),
               [(ref(a["stage"], name), a["started"], a["finished"], a["exit_code"],
                 *(a.get("images_read", {}).get(split, "—") for split in ("train", "validation", "test")))
                for name, a in audits]),
        "## 执行命令", "",
    ]
    records = command_records()
    seen_commands = [record["command"] for record in records]
    records += [
        {"label": a["stage"], "command": shlex.join(a["command"]), "exit_code": a["exit_code"]}
        for _, a in audits if not any(f"--stage {a['stage']}" in command for command in seen_commands)
    ]
    lines.extend(f"- {r['label']}，退出码 {r['exit_code']}。\n\n  `{r['command']}`" for r in records)
    lines += ["", "## 代码验证", ""]
    if (quality := raw("quality")):
        n = quality["crap_at_least_10"]
        d = quality["duplicates_at_least_08"]
        lines += [f"使用覆盖率、Radon 和 AST 相似度执行 ub-mode 质量检查。CRAP ≥10 有 {len(n)} 项，"
                  f"相似度 ≥0.8 有 {len(d)} 对。{ref('质量记录', 'quality.json')}", ""]
    if (mutations := raw("mutations")):
        lines += [f"在临时副本中杀死 {sum(r['killed'] for r in mutations['cases'])}/{len(mutations['cases'])} 个"
                  f"契约变异。{ref('变异记录', 'mutations.json')}", ""]
    return lines


def write_md(path, lines):
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def generate():
    protocol, lock = raw("protocol"), raw("candidates")
    audits = [(name.relative_to(OUTPUT).as_posix(), json_at(name)) for name in (OUTPUT / "audits").glob("*.json")]
    audits.sort(key=lambda entry: (entry[1]["started"], entry[0]))
    lines = [
        "# AIP-VTok C1-Followup 实验报告", "",
        "源提交 `a0f5eb8`，授权的实验分支 `experiment/aip-vtok-c1-followup`。",
        "",
        "## Train-only 校准", "", *render_calibration(raw("calibration"), lock),
        "## Phase 1 · 免训练稠密诊断", "", *render_phase1(raw("validation"), raw("baseline"), raw("phase1_gate")),
        "## Phase 2 / 3 · 受控稠密训练", "", *render_training(raw("training_gate")),
        "## C1-B-Followup 稀疏验收", "", *render_sparse(raw("sparse_gate")),
        "## 保护和回归", "", *render_integrity(raw("protected"), audits),
    ]
    write_md(REPORT, lines)
    write_md(WORK_LOG, work_log(protocol, lock, audits))
