"""Gate 1 report fragments built only from completed, raw experiment artifacts."""

import json

from aip_vtok.evaluation.smoke import BUDGETS


def read(path):
    return json.loads(path.read_text())


def status(directory):
    path = directory / "gate1_decision.json"
    if path.exists():
        gate = read(path)
        if not gate["both_seeds_complete"]:
            return "Gate 1 证据尚不完整"
        return "Gate 1 通过" if gate["passed"] else "Gate 1 熔断，Gate 2/3 未启动"
    if (directory / "gate1_smoke/decision.json").exists():
        return "Gate 1 单视频冒烟已完成，正式双种子阶段进行中"
    return "Gate 1 未运行"


def available_results(directory):
    return [read(path) for path in sorted(directory.glob("gate1_seed*/evaluation.json"))]


def rd_table(seed, metrics):
    rows = [
        f"### seed={seed}",
        "",
        "| K | PSNR 全局 MSE | PSNR 帧均值 | LPIPS-Alex | DISTS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in BUDGETS:
        s = metrics[str(k)]
        rows.append(
            f"| {k} | {s['psnr_global_mse']:.6f} | {s['psnr_frame_mean']:.6f} | "
            f"{s['lpips']:.6f} | {s['dists']:.6f} |"
        )
    return rows + [""]


def quality_section(directory):
    lines = ["## Q2：FSQ 与多预算单调性", ""]
    smoke = directory / "gate1_smoke/decision.json"
    if not smoke.exists():
        return lines + ["未运行 Gate 1，不声称解决塌缩或多预算重建。", ""]
    pilot = read(smoke)
    lines += [
        f"单视频冒烟判定 `{pilot['decision']}`；前后窗口平均损失 "
        f"`{pilot['initial_loss']:.6f} → {pilot['final_loss']:.6f}`。训练帧冒烟不等同于正式验证。",
        "",
    ]
    seeds = available_results(directory)
    if not seeds:
        lines += rd_table("smoke / train", pilot["final_metrics"])
        return lines + ["正式双种子验证尚未完成；不存在跨种子结论。", ""]
    for seed in seeds:
        metrics = seed["summary"]["paired"]
        lines += [
            f"验证帧数 `{metrics['510']['frames']}`，训练更新 `{seed['training_steps']}`；"
            f"独立闭环与同状态配对最大 RGB 误差 `{seed['maximum_independent_rgb_error']}`。",
            "",
        ]
        lines += rd_table(seed["seed"], metrics)
        lines += ["| Pass | 最少 ID 数 | 最低 ID 熵 / bit | 最低逐维熵 / bit |", "|---:|---:|---:|---:|"]
        for pass_id, h in seed["health_summary"].items():
            lines.append(
                f"| {pass_id} | {h['minimum_unique_ids']} | {h['minimum_id_entropy_bits']:.6f} | "
                f"{h['minimum_dimension_entropy_bits']:.6f} |"
            )
        lines += [""]
    decision = directory / "gate1_decision.json"
    if decision.exists():
        gate = read(decision)
        lines += [
            "逐项判定：",
            "",
            "```json",
            json.dumps(gate["checks"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "所有预算均由每个实验路径单次编码得到的同一 1D 序列截取。"
        "I 参考固定使用 2048 Token，P 参考固定使用 510 Token；I 的低显示预算仍需收到参考前缀才能提交。"
        "独立路径实际重建 RGB 后检查逐元素误差为零时才复用相同指标值。",
        "",
    ]
    return lines


def color_section(directory):
    lines = [
        "## Q3：保色",
        "",
        "色度保留率定义为每帧 BT.709 Cb/Cr 空间方差比的均值；"
        "方差比不能独立证明颜色正确，因此同时报告色度 MSE 与有符号均值的绝对偏差。",
        "",
    ]
    seeds = available_results(directory)
    if not seeds:
        return lines + ["正式验证数据未齐，不宣称消除偏灰。", ""]
    lines += ["| Seed | K | 方差保留率 / % | 色度 MSE | 均值偏差 |", "|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for budget in BUDGETS:
            s = seed["summary"]["paired"][str(budget)]
            lines.append(
                f"| {seed['seed']} | {budget} | {s['chroma_retention'] * 100:.4f} | "
                f"{s['chroma_mse']:.8f} | {s['chroma_mean_abs_bias']:.8f} |"
            )
    return lines + [
        "",
        "判定表的 chroma 项只检查预先设定的方差比下界；远大于一的比值不代表颜色正确。"
        "需结合色度 MSE、均值偏差与原尺寸图检查过饱和和伪色。",
        "",
    ]


def intervention_section(directory):
    seeds = available_results(directory)
    if not seeds:
        return []
    lines = [
        "### 后缀干预",
        "",
        "| Seed | 干预 | 平均 MSE 增量 | 按视频重采样 95% CI |",
        "|---:|---|---:|---|",
    ]
    for seed in seeds:
        for name, v in seed["intervention_summary"].items():
            lines.append(
                f"| {seed['seed']} | {name} | {v['mean_mse_increase']:.8f} | "
                f"[{v['ci95_low']:.8f}, {v['ci95_high']:.8f}] |"
            )
    return lines + ["", "区间以独立视频为重采样单位；不把同视频连续帧当成独立样本。", ""]


def diagnosis_section(directory):
    path = directory / "gate1_horizon_diagnostics.json"
    if not path.exists():
        return []
    data = read(path)
    lines = [
        "## 熔断定位",
        "",
        f"以下为 `{data['video']}` 的 `{data['budget']}` Token 诊断，"
        "各模式分别单次编码，I2048/P510 参考规则保持固定。未更新任何模型参数。",
        "",
        "| 模式 | 全局 PSNR / dB | 48–63 帧 PSNR / dB | 第63帧参考潜在 RMS |",
        "|---|---:|---:|---:|",
    ]
    for result in data["results"]:
        lines.append(
            f"| {result['mode']} | {result['summary']['psnr_global_mse']:.6f} | "
            f"{result['late_48_63']['psnr_global_mse']:.6f} | {result['rows'][-1]['reference_latent_rms']:.6f} |"
        )
    baseline = data["results"][0]["rows"]
    a, b, c = baseline[3], baseline[31], baseline[32]
    lines += [
        "",
        f"原路径第 `{a['frame']}` 帧参考潜在 RMS `{a['reference_latent_rms']:.6f}`，"
        f"第 `{b['frame']}` 帧增至 `{b['reference_latent_rms']:.6f}`，"
        f"第 `{c['frame']}` 帧 GOP 重置回 `{c['reference_latent_rms']:.6f}`。"
        f"第 `{b['frame']}` 帧 Anchor 标量达到边界的比例为 `{b['anchor_boundary_fraction'] * 100:.4f}%`，"
        f"最低逐维熵 `{b['anchor_min_dimension_entropy']:.6f}` bit。",
        "",
        "关闭时序预测或提前重置均显著缓解后期失稳，支持预测反馈累积导致状态幅度发散、继而使 FSQ 饱和的解释。"
        "4帧训练窗口内健康的符号分布不能保证GOP32长期稳定；发送与接收同步也不等于递推本身稳定。"
        "GOP4和零预测结果仅用于定位，不替代原协议或质量门槛。",
        "",
    ]
    matched = directory / "gate1_seed1/diagnostics.json"
    if matched.exists():
        d = read(matched)["summary"]["8192"]
        lines += [
            f"相同历史下的前两帧/视频诊断共 `{d['frames']}` 帧："
            f"直通 `{d['matched_direct']['psnr_global_mse']:.6f}` dB，"
            f"FSQ路径 `{d['psnr_global_mse']:.6f}` dB。"
            f"该预算的原始/融合潜在 MSE 为 `{d['raw_latent_mse']:.6f}` / `{d['fused_latent_mse']:.6f}`。"
            "当前表示与原生潜在空间仍有大偏差；RGB损失改善不保证参考投影所需的潜在分布一致性。"
            "投影器的前缀分布适配是待单独检验的原因，不能据此声称唯一归因。",
            "",
        ]
    recheck = directory / "evaluation_recheck.json"
    if recheck.exists():
        check = read(recheck)
        lines += [
            f"种子2初次执行会话在主流程输出完成后报告退出码 `{check['primary_session_exit_code']}`。"
            f"独立复跑比较 `{check['rows']}` 行，CSV全部字段相等 `{check['all_csv_fields_equal']}`，"
            f"汇总JSON相等 `{check['evaluation_json_equal']}`。保留两套原始证据。",
            "",
        ]
    return lines


def engineering_section(directory):
    path = directory / "checkpoint_gpu_replay.json"
    if not path.exists():
        return []
    current = read(path)
    original_path = directory / "checkpoint_gpu_replay_original.json"
    original = read(original_path)
    return [
        "## 工程复核与限制",
        "",
        f"历史训练未启用严格 CUDA 确定性；相同输入/RNG 的参数重放最大差异实测为 "
        f"`{original['maximum_parameter_error']:.10g}`。"
        "新增训练运行时统一配置确定性算法、cuDNN 和 cuBLAS 工作区，并以等价的固定采样替换感知损失全局视图的非确定性插值反传。",
        "",
        f"修复后真实原生 GPU 更新重放：参数最大误差 `{current['maximum_parameter_error']}`，"
        f"损失误差 `{current['loss_error']}`，四流 RNG 一致 `{current['four_rng_states_equal']}`，"
        f"原检查点未改变 `{current['checkpoint_unchanged']}`。"
        "这验证新的续训运行时，不把旧训练追认为从零开始逐位可复现。",
        "",
        "参考前缀在 append 中直接复用新增 Token 的升维结果执行增量 scatter；"
        "不再在到达参考边界时整幅求和。修复前后64帧所有诊断值相等，见 `reference_cache_equivalence.json`。",
        "",
        "Stage 1 当前在参考提交时 detach；4帧为数值闭环采样长度，不能等同于4帧跨时域梯度。"
        "Stage 2 的 TBPTT=2、真正 rANS 分包、Stage 3 的完整实时路径尚未实施。",
        "",
        "![训练曲线](../results/aip_vtok/figures/training_curves.png)",
        "",
        "![174帧多预算质量](../results/aip_vtok/figures/gate1_quality_curves.png)",
        "",
        "![长序列诊断](../results/aip_vtok/figures/horizon_diagnostics.png)",
        "",
        "![原尺寸裁剪对照](../results/aip_vtok/figures/car-roundabout_seed1_crop_comparison.png)",
        "",
    ]
