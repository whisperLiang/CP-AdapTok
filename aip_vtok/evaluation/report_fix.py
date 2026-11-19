"""Render Gate 1-Fix evidence without regenerating the sealed historical sections."""

from statistics import mean

import numpy as np

from aip_vtok.evaluation.fix import frame_metrics, summarize_validation
from aip_vtok.evaluation.fix_gate import (
    HIGH,
    assess_fix_a,
    assess_fix_b,
    assess_seed,
    assess_smoke,
    historical,
    horizon_summary,
    matched_summary,
)
from aip_vtok.evaluation.fix_metrics import pooled_fsq
from aip_vtok.evaluation.formal import summarize_interventions
from aip_vtok.runtime import ROOT, file_hash, write_json
from aip_vtok.training.fix import CONFIG, RESULTS, directories, fix_settings, read_json

BEGIN = "<!-- BEGIN GENERATED GATE 1-FIX -->"
END = "<!-- END GENERATED GATE 1-FIX -->"
LABELS = {"smoke": "Fix-A 冒烟", "fix_a": "Fix-A", "b0": "B0 冻结", "b1": "B1 联合微调"}
MODES = {"paired": "174 帧同状态配对", "independent_64": "64 帧独立闭环", "late_48_63": "独立闭环第 48–63 帧"}


def optional_json(path):
    return read_json(path) if path.is_file() else None


def link(label, path):
    return f"[{label}](../{path.relative_to(ROOT)})"


def experiment(phase, seed):
    directory, _ = directories(phase, seed)
    if not (directory / "evaluation.json").is_file():
        return None
    case = {
        "directory": str(directory.relative_to(ROOT)),
        "inputs": read_json(directory / "inputs.json")["config"],
        "evaluation": read_json(directory / "evaluation.json"),
        "diagnostics": matched_summary(directory),
        "horizon": horizon_summary(directory),
    }
    if phase != "smoke":
        case["validation"] = summarize_validation(frame_metrics(directory))
        case["fsq"] = pooled_fsq(read_json(directory / "health.json"))
        training = read_json(directory / "training.json")
        case["training"] = {
            "source_count": len({row["source"] for row in training["history"]}),
            "peak_allocated_mib": training["peak_allocated_mib"],
        }
    if phase == "fix_c":
        case["interventions"] = summarize_interventions(read_json(directory / "interventions.json"))
        case["curve"] = {
            str(read_json(path)["step"]): summarize_validation(frame_metrics(path.parent))["paired"]
            for path in sorted(directory.glob("step_*/evaluation.json"))
        }
    oracle = optional_json(directory / "tokenizer_oracle.json")
    if oracle:
        case["oracle"] = oracle
    return case


def collect(path):
    cases = {phase: experiment(phase, 1) for phase in LABELS}
    cases.update({f"seed{seed}": experiment("fix_c", seed) for seed in (1, 2)})
    smoke = (
        assess_smoke(directories("smoke", 1)[0], fix_settings("smoke", 1, path)) if cases["smoke"] else None
    )
    fix_a = (
        assess_fix_a(directories("fix_a", 1)[0], fix_settings("fix_a", 1, path)) if cases["fix_a"] else None
    )
    fix_b = (
        assess_fix_b(fix_settings("b0", 1, path)) if (RESULTS / "fix_b/fork_audit.json").is_file() else None
    )
    seeds = {
        key: assess_seed(ROOT / case["directory"], case["inputs"])
        for key, case in cases.items()
        if key.startswith("seed") and case
    }
    integrity = optional_json(RESULTS / "integrity.json")
    tests = optional_json(RESULTS / "tests/verification.json")
    complete = len(seeds) == 2
    required = (smoke, fix_a, fix_b, integrity, tests, *seeds.values())
    gate2 = complete and all(case and case["passed"] for case in required)
    return {
        "historical": historical(),
        "cases": cases,
        "smoke": smoke,
        "fix_a": fix_a,
        "fix_b": fix_b,
        "seed_checks": seeds,
        "integrity": integrity,
        "tests": tests,
        "both_seeds_complete": complete,
        "gate2_allowed": gate2,
    }


def status(result):
    if result["gate2_allowed"]:
        return "Gate 1-Fix 双种子通过，可以另行评估进入 Gate 2"
    if result["smoke"] and not result["smoke"]["passed"]:
        return "Fix-A 在冒烟止损门处停止，Gate 2 暂缓"
    if result["fix_a"] and not result["fix_a"]["passed"]:
        return "Fix-A 触发止损，Gate 2 暂缓"
    if result["fix_b"] and not result["fix_b"]["passed"]:
        return "Fix-B 核验未通过，Gate 2 暂缓"
    if result["both_seeds_complete"]:
        return "Gate 1-Fix 验收未通过，Gate 2 暂缓"
    return "Gate 1-Fix 的阶段证据尚未齐备，Gate 2 暂缓"


def smoke_lines(result):
    smoke = result["smoke"]
    if smoke is None:
        return ["### Fix-A 冒烟", "", "待测。", ""]
    case = result["cases"]["smoke"]
    directory = ROOT / case["directory"]
    training = read_json(directory / "training.json")
    history = training["history"]
    saturation = [
        r["tanh_saturation_ratio"]
        for step in history
        for r in step["records"]
        if "tanh_saturation_ratio" in r
    ]
    first, last = smoke["overfit_k8192_psnr_initial_final"]
    loss_first, loss_last = smoke["loss_first_last10"]
    advisory = case["inputs"]["fix_limits"]["smoke_tanh_saturation_advisory"]
    rows = [
        "### Fix-A 单视频过拟合",
        "",
        f"`{case['inputs']['stage1']['video']}` 原生 1080p、4 帧逐帧提交、{len(history)} 次更新。"
        f"K8192 训练片段 PSNR 从 {first:.3f} 提升到 {last:.3f} dB，"
        f"首末 10 步平均总损失由 {loss_first:.4f} 降至 {loss_last:.4f}。"
        f"GPU 峰值分配 {training['peak_allocated_mib']:.0f} MiB。 "
        f"{link('逐步训练记录', directory / 'training.json')}",
        "",
        f"真实 tanh 饱和比例均值 {mean(saturation):.2%}、P95 {np.quantile(saturation, 0.95):.2%}、"
        f"最大 {max(saturation):.2%}。预设严重饱和提示值 {advisory:.0%}。",
        "",
        "下表取各检查步 I 帧低预算的未加权分项值与加权梯度范数。RGB 项与融合锚定在融合潜在处取梯度，"
        "原始锚定在未融合潜在处取梯度，参考项在固定参考前缀处取梯度。冻结 Projector 的参数梯度为零。",
        "",
        "| 步 | 分项 | 损失 | 加权梯度范数 |",
        "|---:|---|---:|---:|",
    ]
    for update in history:
        record = update["records"][0]
        if "weighted_grad_norm_at_latent" in record:
            terms = record["components"] | {"reference": record["reference_loss"]}
            for key, norm in record["weighted_grad_norm_at_latent"].items():
                rows.append(f"| {update['step']} | {key} | {terms[key]:.6f} | {norm:.3e} |")
    return [*rows, ""]


def attribution_lines(result):
    lines = [
        "### Fix-A 与 Fix-B 的归因对照",
        "",
        "| 实验 | 训练步 | 同历史 K8192 PSNR / dB | 融合潜在 MSE | Δ Fusion MSE | 174 帧 K8192 PSNR / dB | 64 帧 K8192 PSNR / dB | 晚期 PSNR / dB | 晚期参考 L1 | 31 / 63 帧参考潜在 RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    baseline = result["historical"]
    old_delta = baseline["fused_latent_mse"] - baseline["raw_latent_mse"]
    lines.append(
        f"| 508a97e 原始外环 | — | {baseline['matched_psnr']:.3f} | {baseline['fused_latent_mse']:.4f} | "
        f"{old_delta:+.4f} | — | {baseline['horizon']['original_gop32']:.3f} | — | — | — |"
    )
    lines.append(
        f"| 508a97e 零外环预测回放 | — | — | — | — | — | {baseline['horizon']['zero_prediction_gop32']:.3f} | — | — | — |"
    )
    for key, label in LABELS.items():
        case = result["cases"][key]
        if not case:
            lines.append(f"| {label} | 未运行 | — | — | — | — | — | — | — | — |")
            continue
        matched, h = case["diagnostics"][HIGH], case["horizon"]
        val = case.get("validation", {}).get("paired", {}).get(HIGH)
        val_psnr = f"{val['psnr_global_mse']:.3f}" if val else "—"
        rms = " / ".join(f"{h['endpoints'][k]['reference_latent_rms']:.3f}" for k in ("31", "63"))
        lines.append(
            f"| {label} | {case['evaluation']['step']} | {matched['psnr_global_mse']:.3f} | "
            f"{matched['fused_latent_mse']:.4f} | {matched['delta_fusion_mse']:+.4f} | {val_psnr} | "
            f"{h['all']['psnr_global_mse']:.3f} | {h['late_48_63']['psnr_global_mse']:.3f} | "
            f"{h['late_native_feature_l1']:.5f} | {rms} |"
        )
    lines.append("")
    if result["fix_a"]:
        a, case = result["fix_a"], result["cases"]["fix_a"]
        limits, h = case["inputs"]["fix_limits"], case["horizon"]
        endpoints = h["endpoints"]
        verdict = "通过" if a["passed"] else "止损"
        lines.extend(
            [
                f"Fix-A 覆盖 {case['training']['source_count']} 个原生 1080p 训练视频，判定为{verdict}。"
                f"同历史 K8192 PSNR 相对历史变化 {a['matched_psnr_gain_db']:+.3f} dB，"
                f"融合潜在 MSE 为历史值的 {a['matched_fused_mse_fraction']:.1%}。"
                f"预设判据要求 PSNR 增益至少 {limits['minimum_fix_a_psnr_gain_db']:.1f} dB，"
                f"融合 MSE 至多为历史值的 {limits['maximum_fix_a_fused_mse_fraction']:.0%}。"
                f"64 帧 K8192 PSNR 相对历史零预测回放变化 "
                f"{h['all']['psnr_global_mse'] - baseline['horizon']['zero_prediction_gop32']:+.3f} dB。"
                f"第 31/63 帧同历史原生对齐潜在 RMS 为 "
                f"{endpoints['31']['native_aligned_latent_rms']:.3f} / {endpoints['63']['native_aligned_latent_rms']:.3f}。",
                "",
            ]
        )
    if result["fix_b"]:
        b = result["fix_b"]
        b1 = ", ".join(key for key, passed in b["b1_wins"].items() if passed) or "无"
        lines.extend(
            [
                f"B0 与 B1 共用 Fix-A 检查点、图像窗口、增强计划和四流 RNG。"
                f"B1 优于 B0 的比较项为 `{b1}`。选定分支 `{b['winner']}`。 "
                f"{link('Fix-B 分叉审计', RESULTS / 'fix_b/fork_audit.json')}",
                "",
            ]
        )
    return lines


def validation_lines(label, case):
    lines = []
    for mode, budgets in case["validation"].items():
        lines.extend(
            [
                f"#### {label} · {MODES[mode]}",
                "",
                "| K | 全局 PSNR / dB | 帧均 PSNR / dB | LPIPS-Alex | DISTS | 色方差保留率 | 色度 MSE | 色度均值偏差 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for k, row in budgets.items():
            lines.append(
                f"| {k} | {row['psnr_global_mse']:.4f} | {row['psnr_frame_mean']:.4f} | {row['lpips']:.5f} | "
                f"{row['dists']:.5f} | {row['chroma_var_retention_pct']:.2f}% | {row['chroma_mse']:.6f} | "
                f"{row['chroma_mean_bias']:.6f} |"
            )
        lines.append("")
    directory = ROOT / case["directory"]
    lines.extend(
        [
            f"{link('原生逐帧 CSV', directory / 'frame_metrics.csv')} · {link('64 帧漂移曲线原始数据', directory / 'horizon.json')}",
            "",
        ]
    )
    return lines


def matched_lines(label, case, baseline):
    paired = case["diagnostics"][HIGH]
    direct = paired["matched_direct"]["psnr_global_mse"]
    fsq = paired["psnr_global_mse"]
    old_gap = baseline["matched_direct_psnr"] - baseline["matched_psnr"]
    gap = direct - fsq
    direction = "收窄" if gap <= old_gap else "扩大"
    rows = [
        f"#### {label} · 同历史直通与三段 FSQ",
        "",
        f"每个验证视频取前两帧。原生对齐潜在直通为 {direct:.3f} dB，K8192 FSQ 为 {fsq:.3f} dB。"
        f"缺口 {gap:.3f} dB，历史缺口 {old_gap:.3f} dB，较历史{direction} {abs(old_gap - gap):.3f} dB。"
        f"以历史直通 {baseline['matched_direct_psnr']:.3f} dB 为固定参照，当前缺口为 "
        f"{baseline['matched_direct_psnr'] - fsq:.3f} dB。",
        "",
        "| 累计阶段 | K | PSNR / dB | 未融合 MSE | 融合 MSE | Δ Fusion MSE | 相对上一阶段 PSNR / dB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    endpoints = [r for r in case["diagnostics"].values() if r["pass_endpoint"] is not None]
    previous = None
    for r in endpoints:
        current = r["psnr_global_mse"]
        gain = "—" if previous is None else f"{current - previous:+.3f}"
        rows.append(
            f"| Pass {r['pass_endpoint']} | {r['budget']} | {current:.3f} | {r['raw_latent_mse']:.5f} "
            f"| {r['fused_latent_mse']:.5f} | {r['delta_fusion_mse']:+.5f} | {gain} |"
        )
        previous = current
    first = endpoints[0]["fused_latent_mse"]
    final = endpoints[-1]["fused_latent_mse"]
    directory = ROOT / case["directory"]
    rows.extend(
        [
            "",
            f"Pass 1/2 累计后仍保留 Pass 0 融合潜在误差的 {final / first:.1%}。 "
            f"{link('同历史逐帧对照', directory / 'diagnostics.json')}",
            "",
        ]
    )
    return rows


def oracle_lines(case):
    oracle = case.get("oracle")
    if not oracle:
        return []
    rows = [
        "同历史下，逐一把每个 Pass 的 FSQ 值替换为当前冻结 lift 的无范围限制最优六维系数。"
        "评估该 Pass 时沿用前面各 Pass 的实测 FSQ 表示及原有提交历史。",
        "",
        "| Pass | 潜在覆盖 | Pass 前 MSE | 实际 FSQ 后 MSE | 连续六维最小 MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    means = []
    for p in range(3):
        subset = [r for r in oracle["rows"] if r["pass"] == p]
        coverage, before, fsq, continuous = (
            mean(r[key] for r in subset)
            for key in ("active_fraction", "raw_mse_before", "raw_mse_after_fsq", "raw_mse_after_oracle")
        )
        means.append((before, fsq, continuous))
        rows.append(f"| {p} | {coverage:.1%} | {before:.5f} | {fsq:.5f} | {continuous:.5f} |")
    p0, _, p2 = means
    p0_fraction = p0[2] / p0[0]
    p2_fraction = p2[2] / p2[1]
    attribution = (
        "在实测前两段条件下，Pass 2 的已学习六维子空间与采样覆盖限制了剩余残差的表达。"
        if p2_fraction > 0.5
        else "在实测前两段条件下，Pass 2 的实际编码与其连续六维投影之间仍有较大误差。"
    )
    rows.extend(
        [
            "",
            f"Pass 0 的连续投影保留 {p0_fraction:.1%} 的原始潜在误差。冻结 lift 后，"
            f"优化 Pass 2 的六维系数仍保留其实测量化后误差的 {p2_fraction:.1%}。{attribution} "
            f"{link('六帧最小二乘原始记录', ROOT / case['directory'] / 'tokenizer_oracle.json')}",
            "",
        ]
    )
    return rows


def vector(values):
    return " / ".join(f"{value:.3f}" for value in values)


def health_lines(label, case):
    rows = [
        f"#### {label} · FSQ 全验证集分布",
        "",
        "先聚合 174 帧的离散计数再计算熵与有效 ID 数。单帧分位数按帧计算，分开记录离散边界码与真实 tanh 饱和。",
        "",
        "| Pass | 边界码 | tanh 饱和（聚合 / P95 / 最大） | 有效 ID（聚合 / P05 / 最小） | 各维熵 bit（聚合） | 各维熵 bit（帧 P05） | 各维熵 bit（帧最小） |",
        "|---|---:|---|---|---|---|---|",
    ]
    for p, stats in case["fsq"].items():
        pooled, frames = stats["pooled"], stats["per_frame"]
        sat, ids, ent = [
            frames[key] for key in ("tanh_saturation_ratio", "effective_ids", "dim_entropy_bits")
        ]
        rows.append(
            f"| {p} | {pooled['boundary_code_ratio']:.2%} | {pooled['tanh_saturation_ratio']:.2%} / {sat['p95']:.2%} / {sat['max']:.2%} "
            f"| {pooled['effective_ids']:.2f} / {ids['p05']:.2f} / {ids['min']:.2f} "
            f"| {vector(pooled['dim_entropy_bits'])} | {vector(ent['p05'])} | {vector(ent['min'])} |"
        )
    return [*rows, "", link("FSQ 原始计数", ROOT / case["directory"] / "health.json"), ""]


def seed_lines(result):
    b = result["fix_b"]
    rows = ["### Fix-C 双种子验收", ""]
    if b and b["passed"]:
        rows.extend(
            [
                f"`rollout_8f_detached_ref` 固定 `{b['winner']}` 的 Projector 策略。两种子分别从随机初始化的 "
                "Tokenizer/Fusion 与 Gate 0 Projector 起训，实际闭环每帧提交且梯度逐帧截断。",
                "",
            ]
        )
    checks = result["seed_checks"]
    if not checks:
        stopped = (result["fix_a"] and not result["fix_a"]["passed"]) or (b and not b["passed"])
        state = "上游止损，未运行" if stopped else "待测"
        return [
            *rows,
            "| 随机种子 | 五预算正式验收 |",
            "|---|---|",
            f"| 1 | {state} |",
            f"| 2 | {state} |",
            "",
        ]
    rows.extend(["| 随机种子 | 达标 | 未达标项 | K8192 LPIPS 进阶目标 |", "|---|---|---|---|"])
    for key, seed in checks.items():
        failures = "、".join(name for name, passed in seed["checks"].items() if not passed)
        rows.append(f"| {key} | {seed['passed']} | {failures or '—'} | {seed['advanced_lpips']} |")
    rows.append("")
    for key in checks:
        case = result["cases"][key]
        curve = case["curve"] | {str(case["evaluation"]["step"]): case["validation"]["paired"]}
        rows.extend(
            [
                f"#### {key} · 训练检查点与内容干预",
                "",
                "| 步 | 174 帧 K8192 全局 PSNR / dB | LPIPS-Alex | DISTS |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {step} | {v[HIGH]['psnr_global_mse']:.4f} | {v[HIGH]['lpips']:.5f} | {v[HIGH]['dists']:.5f} |"
                    for step, v in curve.items()
                ],
                "",
                "| 干预 | 平均 MSE 增量 | 95% CI（按视频重采样） |",
                "|---|---:|---|",
                *[
                    f"| {kind} | {r['mean_mse_increase']:.7f} | [{r['ci95_low']:.7f}, {r['ci95_high']:.7f}] |"
                    for kind, r in case["interventions"].items()
                ],
                "",
            ]
        )
    return rows


def reconstruction_links(result):
    rows = ["### 原尺寸重建与固定窗口裁剪", ""]
    for key, case in result["cases"].items():
        if not case:
            continue
        directory = ROOT / case["directory"] / "images"
        if directory.is_dir():
            original = directory / "car-roundabout_f063_original_crop.png"
            decoded = directory / "car-roundabout_f063_k8192_crop.png"
            rows.append(
                f"- {LABELS.get(key, key)}。{link('全尺寸 PNG 与 256×256 裁剪', directory)} · "
                f"{link('第 63 帧原图裁剪', original)} · {link('第 63 帧 K8192 裁剪', decoded)}"
            )
        montage = RESULTS / "figures" / f"{key}_car-roundabout_f063_crop_comparison.png"
        if montage.is_file():
            rows.append(f"- {LABELS.get(key, key)}。{link('第 63 帧五档预算像素等比对照', montage)}")
    return [*rows, ""]


def build_figures(result):
    from aip_vtok.evaluation.figures import comparison, fix_horizon_plot

    cases = {key: ROOT / case["directory"] for key, case in result["cases"].items() if case}
    if not cases:
        return {}
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    horizons = {key: directory / "horizon.json" for key, directory in cases.items()}
    plot = output / "horizon.png"
    fix_horizon_plot(horizons, plot)
    inputs = list(horizons.values())
    for key, directory in cases.items():
        label = "car-roundabout_f063"
        target = output / f"{key}_{label}_crop_comparison.png"
        inputs.extend(comparison(directory / "images", target, label, crop=True))
    provenance = {str(path.relative_to(ROOT)): file_hash(path) for path in inputs}
    write_json(output / "sources.json", provenance)
    return {"horizon": str(plot.relative_to(ROOT))}


def render_report(result):
    lines = [
        "## Gate 1-Fix",
        "",
        f"**{status(result)}。**",
        "",
        "Fix-A 使用当前闭环的 DCVC 对齐潜在作为锚定目标，取消外部 TemporalPredictor。"
        "同一未融合的 I2048/P510 前缀同时用于监督与实际提交。参考损失穿过冻结 Projector 回传，"
        "下一帧通过已截断梯度的提交状态展开。Fusion 沿用 130 通道零初始化残差结构。",
        "",
        *smoke_lines(result),
        *attribution_lines(result),
        *seed_lines(result),
    ]
    for key, case in result["cases"].items():
        if case:
            label = LABELS.get(key, key)
            if "validation" in case:
                lines.extend(validation_lines(label, case))
                lines.extend(health_lines(label, case))
            lines.extend(matched_lines(label, case, result["historical"]))
            lines.extend(oracle_lines(case))
    lines.extend(reconstruction_links(result))
    if result["figures"]:
        lines.extend([link("64 帧参考 RMS 与逐帧 PSNR 曲线", ROOT / result["figures"]["horizon"]), ""])
    audit = result["integrity"]
    if audit:
        source = audit["projector_w128"]
        lines.extend(
            [
                "### 隔离校验",
                "",
                f"Gate 0 原始 Projector SHA256 `{source['after']}`，与运行前及 508a97e 封存清单一致"
                f"的核验结果为 `{audit['passed']}`。 "
                f"{link('完整审计', RESULTS / 'integrity.json')}",
                "",
            ]
        )
    tests = result["tests"]
    if tests:
        lines.extend(
            [
                f"联合回归 {tests['tests']} 项，失败 {tests['failures']} 项，错误 {tests['errors']} 项，跳过 {tests['skipped']} 项。 "
                f"{link('测试核验', RESULTS / 'tests/verification.json')}",
                "",
            ]
        )
    return lines


def render_log(result):
    rows = [
        "## Gate 1-Fix",
        "",
        "- 按 `ub-mode` 核对 508a97e 的 Fusion 输入、固定参考前缀与状态所有权，再用失败测试覆盖真实 tanh、双边色度和冻结 Projector 参考梯度。",
        "- 新模型以 `gate1-fix-v2` 保存。旧版入口显式实例化历史预测器，供历史检查点复验。",
        "- 引入原生同历史潜在与参考目标，保留确定性 CUDA、四流 RNG 和真实三段残差。",
        "",
    ]
    for key, case in result["cases"].items():
        if not case:
            continue
        directory = ROOT / case["directory"]
        meta = case["evaluation"]
        rows.append(
            f"- {LABELS.get(key, key)} 完成 {meta['step']} 步。训练检查点 SHA256 `{meta['checkpoint_sha256']}`。 "
            f"{link('评估记录', directory / 'evaluation.json')}"
        )
        if "oracle" in case:
            rows.append("- 对同历史的三个冻结 lift 分别求最小二乘六维系数，测量当前 Tokenizer 的表示上限。")
    if result["tests"]:
        rows.append(
            "- 在评估与回归共享 GPU 时旧版 CUDA 流测试出现一次失败。单项复跑和独占 GPU 联合回归均通过。 "
            f"{link('回归记录', RESULTS / 'tests/verification.json')}"
        )
    quality = RESULTS / "tests/quality_audit.json"
    if quality.is_file():
        rows.append(
            "- 使用仓库内的 Radon、覆盖率、AST 相似度和行为变异工具检查本轮改动。覆盖率按实际插桩的入口记录。 "
            f"{link('工程核验', quality)}"
        )
    rows.extend(
        [
            "",
            f"**{status(result)}。**",
            "",
            "### 复现",
            "",
            "```bash",
            "python -m aip_vtok.training.fix smoke",
            "python -m aip_vtok.evaluation.fix smoke",
            "python -m aip_vtok.evaluation.fix_gate smoke",
            "python -m aip_vtok.training.fix fix_a",
            "python -m aip_vtok.evaluation.fix fix_a",
            "python -m aip_vtok.evaluation.fix_gate fix_a",
            "python -m aip_vtok.evaluation.report --gate1-fix",
            "```",
            "",
        ]
    )
    return rows


def replace_section(path, lines):
    original = path.read_text()
    head, separator, tail = original.partition(BEGIN)
    if separator:
        _, end, tail = tail.partition(END)
        if not end:
            raise ValueError("unterminated Gate 1-Fix report section")
    else:
        head = original + "\n"
    updated = f"{head}{BEGIN}\n\n" + "\n".join(lines) + f"\n{END}" + (tail if separator else "\n")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(updated)
    temporary.replace(path)


def build_fix_report(path=CONFIG):
    result = collect(path)
    result["figures"] = build_figures(result)
    write_json(RESULTS / "summary.json", {**result, "status": status(result)})
    replace_section(ROOT / "docs/aip_vtok_report.md", render_report(result))
    replace_section(ROOT / "docs/aip_vtok_work_log.md", render_log(result))
    return result
