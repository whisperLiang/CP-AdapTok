"""Verify the parked pilot and generate its numerical report from the evidence."""

import csv
from itertools import product
from statistics import median

from aip_vtok.c1.scan import BUDGETS
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.sparse_rfsq.protocol import FOCUS, UPDATES

from .evaluation import CLOSED, FIXED
from .latency import LATENCY
from .models import ARMS
from .phase0_report import quality, table
from .phase1 import ARCHITECTURES, require_phase1
from .pilot import COMPLETE, EVIDENCE, METHODS, completed_training
from .protocol import BASE, BRANCH, OUTPUT, PROVENANCE

KEYS = ("psnr_global_mse", "psnr_frame_mean", "lpips", "dists")
REPORT = OUTPUT / "final_report.md"


def read_metrics():
    with (OUTPUT / "pilot_metrics.csv").open() as handle:
        rows = [{**row, **{k: float(row[k]) for k in KEYS}, "budget": int(row["budget"])}
                for row in csv.DictReader(handle)]
    index = {(r["scope"], r["history"], r["arm"], r["budget"], r["frame_type"]): r for r in rows}
    if len(index) != len(rows) or len(rows) != 400:
        raise RuntimeError("pilot metric inventory requires all 400 cohort and I/P aggregates")
    return index


def quality_table(metrics, scope, budgets):
    rows = ((history, k, METHODS[arm], *quality(metrics[scope, history, arm, k, "all"]))
            for history, arm, k in product((FIXED, CLOSED), METHODS, budgets))
    return table(("历史", "K", "方案", "全局 PSNR dB", "帧均 PSNR dB", "LPIPS", "DISTS"), rows)


def focus_delta_table(metrics):
    rows = []
    for mode, k in product((FIXED, CLOSED), BUDGETS):
        base = metrics["validation", mode, "E3", k, "all"]
        sld = metrics["validation", mode, "E4", k, "all"]
        rows.append((mode, k, *(f"{sld[m] - base[m]:+.5f}" for m in (KEYS[0], *KEYS[2:]))))
    return table(("历史", "K", "SLD − R-Wide PSNR dB", "Δ LPIPS", "Δ DISTS"), rows)


def feedback_table(decisions):
    rows = []
    for arm, record in decisions.items():
        measurements = record["gains"]["late"]
        gains = {(r["history"], r["budget"]): r for r in measurements}
        for k in BUDGETS:
            fixed, closed = gains[FIXED, k], gains[CLOSED, k]
            delta = (closed[key] - fixed[key] for key in ("psnr_gain_db", "lpips_change", "dists_change"))
            rows.append((METHODS[arm], k, *(f"{d:+.5f}" for d in delta)))
    return table(("方案", "K", "PSNR 增益变化 dB", "LPIPS 差值变化", "DISTS 差值变化"), rows)


def gradient_rows(training, decisions):
    for arm, data in training.items():
        probes = [r["gradient_diagnostics"] for r in data["updates"] if r["gradient_diagnostics"] is not None]
        ratios = [g["weighted_lpips_to_mse"] for g in probes if g["weighted_lpips_to_mse"] is not None]
        cosines = [g["cosines"]["lpips/anchor"] for g in probes if g["cosines"]["lpips/anchor"] is not None]
        late = [r for r in decisions[arm]["gains"]["late"] if r["history"] == CLOSED]
        yield (METHODS[arm], f"{median(ratios):.3f}", f"{min(ratios):.3f}–{max(ratios):.3f}",
               f"{sum(c < 0 for c in cosines)}/{len(cosines)}",
               f'{max(r["dists_change"] for r in late):+.5f}', data["clipped_updates"])


def training_section(training, decisions):
    loss_rows = []
    for arm, data in training.items():
        holdout = data["train_holdout"]
        tail = data["updates"][-50:]
        average = sum(r["loss"] for r in tail) / len(tail)
        loss_rows.append((METHODS[arm], data["steps"], f'{data["seconds"]:.1f}', f"{average:.5f}",
                          *(f'{holdout[str(n)]["2048"]["psnr_global_mse"]:.3f}' for n in (0, 100, 300, 500))))
    return [
        "## 训练与梯度",
        "E2/E3/E4 共用 W1 的 Train 划分、500 个更新的实际监督帧和裁剪种子。每次以首帧启动 I/P/P/P，"
        "统一复用 W1 的按帧型预算采样和损失。AdamW 的学习率为 0.001，梯度裁剪阈值为 1.0。"
        "损失权重为 L1 1.0、MSE 1.0、BT.709 色度 0.20、LPIPS 0.15、anchor 0.05、reference 0.15。"
        "W1 的原始窗口计划含最长 26 帧前置因果历史。E1 的权重用于本轮同条件评估，训练参照保留历史差异。",
        "下面的收敛指标使用冻结 Train holdout 的 6 个四帧窗口。训练损失为最后 50 步平均值。",
        table(("方案", "更新", "更新用时 s", "末 50 步损失", "K2048 @0", "@100", "@300", "@500"), loss_rows),
        "每 50 步以恢复主体参数计算三个原始梯度范数、乘权重后的范数和两两余弦。"
        "晚期 DISTS 来自同一条 Validation 轨迹。每个架构仅有一个种子，梯度与 DISTS 的并列比较只支持描述性判断。",
        table(("方案", "加权 LPIPS/MSE 中位数", "范围", "LPIPS/Anchor 负余弦", "最坏晚期 ΔDISTS", "裁剪更新"),
              gradient_rows(training, decisions)),
        "加权梯度比的排序与最坏晚期 DISTS 增量的排序不同。R48-G 与 R-Wide 的 LPIPS/Anchor 负余弦占比相同，"
        "晚期 DISTS 恶化幅度有明显差异。各模型主体参数形状也不同，绝对梯度范数应结合本模型损失分量解读。",
        "[冻结训练计划](pilot_plan.json) · [E2 训练](pilot_train_E2.json) · [E3 训练](pilot_train_E3.json) · "
        "[E4 训练](pilot_train_E4.json)",
    ]


def gate_section(decisions):
    def status(passed):
        return "通过" if passed else "失败"

    rows = ((METHODS[arm], *(status(g["passed"]) for g in record["gates"].values()), status(record["passed"]))
            for arm, record in decisions.items())
    late = []
    for arm, record in decisions.items():
        for r in record["gains"]["late"]:
            if r["history"] == CLOSED:
                late.append((METHODS[arm], r["budget"], f'{r["psnr_gain_db"]:+.3f}',
                             f'{r["mean_frame_psnr_gain_db"]:+.3f}', f'{r["lpips_change"]:+.5f}',
                             f'{r["dists_change"]:+.5f}', f'{r["rms"]:.4f}'))
    return [
        "## 原 S2 门控",
        table(("方案", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "总判定"), rows),
        "门限和判定函数沿用原 S2。G3 同时要求全局与帧均 PSNR 降幅不超过 0.10 dB。"
        "G4 对全部五档要求 LPIPS 和 DISTS 增量均不超过 +0.005。"
        "下表是相对同扫描、同量化器 B2 的 CLOSED 晚期差值。",
        table(("方案", "K", "Δ 全局 PSNR", "Δ 帧均 PSNR", "Δ LPIPS", "Δ DISTS", "参考 RMS"), late),
        "[所有门控的实测值、裕量和阈值](pilot_gates.json)",
    ]


def efficiency_section(latency):
    rows = []
    for case in latency["rows"]:
        if case["frame_type"] != "P" or case["budget"] != 2048:
            continue
        t = case["timing"]
        rows.append((case["network"], case["parameters"], f'{case["recovery_macs"] / 1e6:.2f}',
                     f'{t["receiver_recovery"]["p50_ms"]:.3f}', f'{t["rgb_decoder"]["p50_ms"]:.2f}',
                     f'{t["backbone_encode"]["p50_ms"]:.2f}', f'{t["token_select_quantize"]["p50_ms"]:.3f}',
                     f'{t["end_to_end"]["p50_ms"]:.2f}/{t["end_to_end"]["p95_ms"]:.2f}',
                     f'{case["peak_gpu_allocated_mib"]:.0f}'))
    return [
        "## 效率与负载",
        f'{latency["hardware"]}，原生 1080p，B=1，FP32。每个帧型与预算组合预热 {latency["warmup"]} 次，'
        f'同步计时 {latency["repeats"]} 次。端到端含发送和接收两端参考提交。',
        table(("方案", "参数", "恢复 MMACs", "恢复 p50 ms", "RGB 解码 p50 ms", "编码 p50 ms", "选点量化 p50 ms",
               "端到端 p50/p95 ms", "峰值 MiB"), rows),
        "表中数据来自 P 帧 K2048。MACs 覆盖恢复网络卷积与公共高斯卷积，FLOPs 约为其两倍。"
        "恢复计时还包括整数解码、掩码和画布构造、插值、融合与观测覆盖。",
        "I 帧在 K510 显示时仍需传送 I2048 的参考联合前缀，共 2048 个符号。"
        "P 帧的参考仅需 P510，任何显示档都已包含该前缀。账本使用 16 bit/符号、模拟封装开销和 Train 经验熵估算。",
        "[每个帧型和预算的同步延迟及负载账本](pilot_latency.json)",
    ]


def sld_wins(metrics, mode):
    wins = []
    for metric, direction in (("psnr_global_mse", 1), ("lpips", -1), ("dists", -1)):
        deltas = (metrics["validation", mode, "E4", k, "all"][metric]
                  - metrics["validation", mode, "E3", k, "all"][metric] for k in BUDGETS)
        wins.append(sum(direction * delta > 0 for delta in deltas))
    return "/".join(map(str, wins))


def decision_section(phase0, factorial, metrics, decisions):
    diagnostics = phase0["changes"]
    scans = factorial["validation"]["contrasts"]["all"]
    scan = scans[FIXED]
    def aggregate(arm, k):
        return metrics["validation", FIXED, arm, k, "all"]

    fractions = []
    for k in BUDGETS:
        baseline = aggregate("E0", k)[KEYS[0]]
        max_gain = max(aggregate(arm, k)[KEYS[0]] - baseline for arm in ARMS)
        if max_gain > 0:
            fractions.append((aggregate("E2", k)[KEYS[0]] - baseline) / max_gain)
    g4_margin = {arm: min(min(r["lpips_margin"], r["dists_margin"])
                          for r in record["gates"]["G4"]["by_budget"]) for arm, record in decisions.items()}
    closest = max(g4_margin, key=g4_margin.__getitem__)
    passed = [arm for arm, d in decisions.items() if d["passed"]]
    all_late_dists_fail = all(any(r["dists_margin"] < 0 for r in d["gates"]["G4"]["by_budget"])
                              for d in decisions.values())
    if passed:
        next_model = METHODS[max(passed, key=g4_margin.__getitem__)]
        next_round = f"只将 {next_model} 放行至下一轮 2000-step 双种子验证。"
    elif all_late_dists_fail:
        psnr_passes = [arm for arm, d in decisions.items() if all(d["gates"][g]["passed"] for g in ("G1", "G3"))]
        candidate = max(psnr_passes or list(decisions), key=g4_margin.__getitem__)
        next_round = f"只在 {METHODS[candidate]} 上做 LPIPS=0.10 单变量诊断，复用 seed=1、四帧 500 步协议，复查 G2/G4。"
    else:
        next_round = "只在 Train 内验证 K510 的扫描前缀，同时约束 PSNR、LPIPS 和 DISTS，并保持 A2 固定。"
    headroom = (f"R48-G 在正增益档的最佳全局 PSNR 增益中的占比中位数为 {median(fractions):.0%}。"
                if fractions else "三种新模型在 FIXED 的五档均未产生全局 PSNR 正增益。")
    k510 = {mode: metrics["validation", mode, "E4", 510, "all"]["psnr_global_mse"]
                 - metrics["validation", mode, "E3", 510, "all"]["psnr_global_mse"] for mode in (FIXED, CLOSED)}
    return [
        "## 决策",
        f'1. 24D 截断是显著瓶颈，独立配对效应为 {diagnostics["channel"]["all"][KEYS[0]]:+.2f} dB。'
        f'K8192 空间稀疏化的配对效应为 {diagnostics["spatial"]["all"][KEYS[0]]:+.2f} dB，A2 量化也有显著代价。',
        f'2. K510 的 PSNR 差距以扫描效应为主。共享历史下 A2 的扫描效应为 {scan["scan_a2"][KEYS[0]]:+.2f} dB，'
        f'两种扫描下 A2 的量化效应分别为 {scan["quant_old"][KEYS[0]]:+.2f} 和 {scan["quant_sp1"][KEYS[0]]:+.2f} dB。'
        f'独立历史使 Q4−Q1 再变化 {scans["closed_minus_fixed"]["Q4_minus_Q1"][KEYS[0]]:+.2f} dB。'
        "SP-1 仍有 K510 感知指标优势。",
        "3. 以各档 B2 为基线，将 R48-G 的 FIXED PSNR 增益除以 E2/E3/E4 中最佳增益。"
        + headroom + "R48-G 的 CLOSED PSNR 在 K1024 和 K2048 下降。与 W1 的差距还涉及两套训练历史。",
        "4. R-Wide 与 SLD 的容量匹配误差为 0.087%。按 PSNR、LPIPS、DISTS 的顺序，"
        f"SLD 领先 R-Wide 的预算档数在 FIXED 为 {sld_wins(metrics, FIXED)}，"
        f"在 CLOSED 为 {sld_wins(metrics, CLOSED)}。每项均比较五档。",
        "5. 参数匹配对照检验的是 SLD 整体架构的收益，包含多尺度、融合和置信门控。"
        f'SLD 的 K510 PSNR 优势从 FIXED 的 {k510[FIXED]:+.3f} dB 扩大到 CLOSED 的 {k510[CLOSED]:+.3f} dB。'
        "同码流 DISTS 在五档均优于 R-Wide，PSNR 与 LPIPS 则随预算而变化。"
        "现有 Pilot 证据不足以确定优胜架构。",
        f"6. 按五档 LPIPS、DISTS 的最小门槛裕量，{METHODS[closest]} 最接近晚期感知门槛，"
        f"其最小裕量为 {g4_margin[closest]:+.5f}。",
        ("7. 原门控允许已通过的方案进入下一轮。" if passed else
         "7. 原门控要求七项全部通过。本轮停在 500 步，暂缓 2000-step 训练。"),
        ("8. K510 的扫描可单独研究。当前放行条件应优先处理晚期感知失败。"
         if all_late_dists_fail else "8. K510 的扫描优先级适用于低预算 PSNR，同时应保留同码流感知约束。"),
        f"**下一轮唯一建议**　{next_round}",
    ]


def verify_evidence(run, training, evaluation, latency):
    outputs = {name: file_hash(OUTPUT / name) for name in EVIDENCE}
    provenance = file_hash(PROVENANCE)
    if (outputs != evaluation["outputs"] or evaluation["rows"] != 8700
            or provenance != evaluation["provenance_sha256"] or provenance != latency["provenance_sha256"]):
        raise RuntimeError("final evidence does not match the evaluated cohort or frozen provenance")
    with (OUTPUT / "pilot_per_frame.csv").open() as handle:
        if sum(1 for _ in csv.DictReader(handle)) != evaluation["rows"]:
            raise RuntimeError("per-frame CSV is incomplete")
    expected = set(product(METHODS, ("I", "P"), BUDGETS))
    actual = {(r["arm"], r["frame_type"], r["budget"]) for r in latency["rows"]}
    if (actual != expected or len(latency["rows"]) != len(expected)
            or latency["checkpoints_sha256"] != {a: d["checkpoints_sha256"]["500"] for a, d in training.items()}):
        raise RuntimeError("latency evidence lacks a matched budget, frame type or checkpoint")
    audits = [json_at(p) for p in sorted((OUTPUT / "audits").glob("*.json"))]
    if any(a.get("images_read", {}).get("test", 0) or a.get("denied_reads", 0) for a in audits):
        raise RuntimeError("sealed pixel access detected")
    return run.verify_integrity() | {"audited_stages": len(audits), "test_images_read_across_stages": 0}


def run_report(run):
    tests = require_phase1()
    training = completed_training()
    evaluation = json_at(COMPLETE)
    latency = json_at(LATENCY)
    integrity = verify_evidence(run, training, evaluation, latency)
    phase0 = json_at(OUTPUT / "phase0_diagnostics.json")
    factorial = json_at(OUTPUT / "phase0_factorial.json")
    decisions = json_at(OUTPUT / "pilot_gates.json")["decisions"]
    metrics = read_metrics()
    chain = [p for p in phase0["points"] if p["arm"] in ("O0", "O1", "O2") or p["budget"] == 8192]
    lines = [
        "# AIP-VTok Sparse Latent Decoder Pilot",
        f"`{BRANCH}` 从 `{BASE[:7]}` 出发。E2、E3 与 E4 各运行 seed=1 的 {UPDATES} 次更新。"
        f"原生 1080p Validation 覆盖 174 帧及 car-roundabout 64 帧。"
        f'逐帧记录 {evaluation["rows"]} 行，Test 图像读取为 0。',
        "## 无训练瓶颈诊断",
        table(("方案", "K", "全局 PSNR dB", "帧均 PSNR dB", "LPIPS", "DISTS"),
              ((p["arm"], p["budget"] or "连续", *quality(p["all"])) for p in chain)),
        "O0 衡量冻结 RGB 解码器在完整对齐潜在输入下的表现。学习型解码器的理论画质上界需要独立证明。"
        "各项 PSNR 差值是基于全局 MSE 的对数测量，误差分量的线性归因需在误差域另行验证。",
        "[完整 O0–O4、K510 二因子与历史核验](phase0_report.md)",
        "## 同预算质量",
        "FIXED 共用冻结 C1 A1/B2 的 I2048/P510 参考。五个恢复器的展示前缀均来自同一 A2/SP-1 包。"
        "CLOSED 按各方案自身配置独立推进参考，码流随历史变化。每帧在各端分别提交一次参考。",
        "R48-G 在 W1 的 48 宽度主体输入中加入对齐后的 Gaussian base。R-Wide 用单尺度 88 宽度主体。"
        "SLD 融合 68×120 与 34×60 特征，以 sigmoid 门控修正强度。未传送的 104 个通道由模型预测，"
        "已收到的 FSQ 槽位保留整数解码值。",
        quality_table(metrics, "validation", FOCUS),
        "下面比较相近参数量的 SLD 与 R-Wide。负的 LPIPS 或 DISTS 差值表示 SLD 的该项指标更好。",
        focus_delta_table(metrics),
        "[全预算 Validation、64 帧、晚期和 I/P 汇总](pilot_metrics.csv) · [逐帧记录](pilot_per_frame.csv)",
        *training_section(training, decisions),
        *gate_section(decisions),
        "## 因果历史对晚期增益的影响",
        "以下为 car-roundabout 第 48–63 帧中，CLOSED 相对 B2 的差值减去 FIXED 相对 B2 的差值。"
        "该配对差值衡量闭环历史及其后续码流对观测到的架构收益的影响。"
        "三个模型都已在 FIXED 下出现晚期 DISTS 大幅恶化，独立历史改变其幅度。",
        feedback_table(decisions),
        *efficiency_section(latency),
        "## 原生 1080p 重建对照",
        "car-roundabout 第 55 帧（从 0 计数），FIXED。全部图块截取相同原生 RGB 区域。",
        "![相同帧的 Original、B2、W1、R48-G、R-Wide 和 SLD](pilot_reconstruction.png)",
        *decision_section(phase0, factorial, metrics, decisions),
        "## 复现与审计",
        f'联合测试通过 {tests["counts"]["legacy"] + tests["counts"]["new"]} 项，覆盖 T1–T15。'
        f'冻结资产清单 SHA256 为 `{file_hash(PROVENANCE)}`。'
        "SHA256 覆盖原基础权重、Alignment、Projector、W1、划分、两种扫描、校准和感知权重。",
        "[复现命令与工作记录](../../../docs/aip_vtok_sparse_decoder_report.md) · "
        "[测试证据](phase1_tests.json) · [模型结构](phase1_architectures.json) · [输入清单](provenance.json)",
    ]
    REPORT.write_text("\n\n".join(lines) + "\n")
    write_json(OUTPUT / "pilot_verification.json", {
        "branch": BRANCH, "frozen_base": BASE, "integrity": integrity,
        "test_gate_sha256": file_hash(OUTPUT / "phase1_tests.json"), "checkpoints": {
            a: d["checkpoints_sha256"] for a, d in training.items()
        },
        "artifact_sha256": {name: file_hash(OUTPUT / name) for name in (
            *EVIDENCE, COMPLETE.name, LATENCY.name, REPORT.name, ARCHITECTURES.name
        )},
        "cohort": evaluation["cohort"], "rows": evaluation["rows"], "sealed_controls_replay": evaluation["sealed_controls_replay"],
    })
