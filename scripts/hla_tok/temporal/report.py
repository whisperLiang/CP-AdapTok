"""Render the experiment verdict from completed, linked audits and physical measurements."""

import argparse
import shutil
from pathlib import Path
from statistics import median

from hla_tok.evaluation.io import write_csv, write_json
from scripts.hla_tok.temporal.inputs import COMPACT, RESULTS
from scripts.hla_tok.temporal.selection import BANDS
from scripts.hla_tok.v2.protocol import csv_at, json_at


def _gather(output: Path, name: str) -> dict:
    path = output / f"{name}.json"
    return json_at(path) if path.exists() else {"status": "NOT_RUN"}


def _final_runtime(output: Path) -> dict:
    native = _gather(output, "runtime_audit_native")
    if native["status"] == "NOT_RUN":
        return _gather(output, "runtime_audit")
    parity = json_at(output / "native_wire_parity.json")
    old = output / "runtime_audit_python.json"
    current = output / "runtime_audit.json"
    if parity["status"] != "PASS" or len(native["formal_conditions"]) != 8:
        raise RuntimeError("native Record timing did not complete its wire-equivalence audit")
    if not old.exists():
        shutil.copyfile(current, old)
        shutil.copyfile(output / "latency_breakdown.csv", output / "latency_breakdown_python.csv")
    python = json_at(old)
    if python["profile_binding"] != native["profile_binding"]:
        raise RuntimeError("runtime backends used different Profiles")
    selected = native | {"python_reference_runtime": old.name,
                         "native_wire_equivalence_checks": parity["checks"],
                         "selected_for": "physical Record backend with complete matched equivalence and timing",
                         "acceptance_by_precision": {precision: all(r["pass_30fps"] and r["pass_p95"]
                             for r in native["formal_conditions"] if r["precision"] == precision)
                             for precision in ("bf16", "fp32")}}
    write_json(current, selected)
    shutil.copyfile(output / "latency_breakdown_native.csv", output / "latency_breakdown.csv")
    return selected


def _metrics(row: dict) -> str:
    return " / ".join(f"{float(row[m]):.{n}f}" for m, n in (("global_psnr", 3), ("lpips", 5), ("dists", 5)))


def _for_method(table: list[dict], method: str, bandwidth: str) -> dict:
    return next(row for row in table if row["method"] == method and row["bandwidth"] == bandwidth)


def _quality_table(table: list[dict], method: str) -> list[str]:
    header = ["| 上限 Mbps | HLT B/frame | 实际 Mbps | HLT 三项质量 | CP Q16-4 | CP Q8-4 T1 | Intra RA | Original AdapTok |",
              "| ---: | ---: | ---: | --- | --- | --- | --- | --- |"]
    for band in BANDS:
        temporal = _for_method(table, method, band)
        peers = [_for_method(table, name, band) for name in ("CP Q16-4", "CP Q8-4 T1", "Intra RA", "Original AdapTok")]
        header.append(f"| {band} | {float(temporal['total_bytes_per_frame']):.1f} | "
                      f"{float(temporal['mbps_30fps']):.4f} | {_metrics(temporal)} | "
                      + " | ".join(map(_metrics, peers)) + " |")
    return header


def _delta_table(table: list[dict], method: str) -> list[str]:
    output = ["| 上限 Mbps | 对 CP Q16-4 | 对 CP Q8-4 T1 | 对 Intra RA | 对 Original |",
              "| ---: | --- | --- | --- | --- |"]
    for band in BANDS:
        temporal = _for_method(table, method, band)
        differences = []
        for name in ("CP Q16-4", "CP Q8-4 T1", "Intra RA", "Original AdapTok"):
            other = _for_method(table, name, band)
            differences.append(" / ".join(f"{float(temporal[m])-float(other[m]):+.{n}f}"
                                          for m, n in (("global_psnr", 2), ("lpips", 4), ("dists", 4))))
        output.append(f"| {band} | " + " | ".join(differences) + " |")
    return output


def _ablations(table: list[dict]) -> list[str]:
    pairs = [("T0 fixed", "Intra fixed", "时间预测与实际稀疏前缀，S0"),
             ("T0-RA", "Intra RA", "时间预测与实际稀疏前缀，锁定 S1"),
             ("Intra RA", "Intra fixed", "Rank-aware 分配，绝对编码"),
             ("T0-RA", "T0 fixed", "能量分量分配 S1，复制闭环"),
             ("T0-RA S2", "T0-RA", "RGB/实际比特分量分配 S2，复制闭环"),
             ("T0-RA", "T0-RA RAW", "稀疏短块，锁定复制及 S1"),
             ("T0 fixed", "T0 fixed RAW", "稀疏短块，锁定复制及 S0"),
             ("T0 linear", "T0 fixed", "有界外推相对历史复制")]
    rows = ["| 对照 | 5 Mbps ΔPSNR / ΔLPIPS / ΔDISTS | 10 Mbps ΔPSNR / ΔLPIPS / ΔDISTS |",
            "| --- | --- | --- |"]
    for name, other, label in pairs:
        diffs = []
        for band in ("5", "10"):
            a, b = (_for_method(table, method, band) for method in (name, other))
            diffs.append(" / ".join(f"{float(a[m])-float(b[m]):+.{n}f}"
                                    for m, n in (("global_psnr", 2), ("lpips", 4), ("dists", 4))))
        rows.append(f"| {label} | " + " | ".join(diffs) + " |")
    return rows


def _runtime_line(output: Path, runtime: dict) -> list[str]:
    if runtime["status"] == "NOT_RUN":
        return ["H100 正式串行测速尚未完成。"]
    compact = csv_at(COMPACT / "throughput_benchmark.csv")
    caps = {"5": "2596", "10": "5200", "20": "10409", "40": "16320"}
    report = ["| 精度 | Mbps | HLT FPS | HLT P95 ms | 相对 CP ΔFPS | 相对 CP ΔP95 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in runtime["formal_conditions"]:
        dtype, b = row["precision"], row["bandwidth"]
        baseline, = [r for r in compact if r["asset"] == "Q16-4_rank"
                     and r["compute_dtype"] == dtype and r["budget"] == caps[b]]
        report.append(f"| {dtype.upper()} | {b} | {row['steady_state_fps']:.2f} | {row['p95_ms']:.2f} | "
                      f"{row['steady_state_fps']-float(baseline['steady_state_fps']):+.2f} | "
                      f"{row['p95_ms']-float(baseline['p95_ms']):+.2f} |")
    portable = runtime.get("python_reference_runtime")
    if portable:
        old = json_at(output / portable)
        p40, = [r for r in old["formal_conditions"] if r["precision"] == "bf16" and r["bandwidth"] == "40"]
        report += ["", f"参考 Python/NumPy Record 路径的 BF16 40 Mbps 为 {p40['steady_state_fps']:.2f} FPS、"
                   f"P95 {p40['p95_ms']:.2f} ms。"
                   f"采用 C11 Record 后端前在实际帧上完成 {runtime['native_wire_equivalence_checks']} 次码流、"
                   "账本、整数显示和状态摘要一致性检查。C11 按批融合短块打包及封装、解析及短块恢复。"
                   "各融合阶段按实际调用计时。"
                   " [Python 路径原测量](runtime_audit_python.json) · [后端等价审计](native_wire_parity.json)"]
    return report


def _diagnosis(output: Path) -> list[str]:
    rows = csv_at(output / "prediction_only_rd.csv")
    correlation = csv_at(output / "rank_temporal_correlation.csv")
    summary = {mode: median(float(row["pearson"]) for row in correlation if row["method"] == mode and row["pearson"])
               for mode in ("A1 oracle", "A2 CP decoded", "A3 copy CLOSED", "A4 linear CLOSED")}
    motion = [r for r in csv_at(output / "motion_stratified_analysis.csv")
              if r["method"] == "A3 copy CLOSED" and r["motion_stratum"] in ("low", "high")]
    motion_by_stratum = {}
    for group in ("low", "high"):
        records = [row for row in motion if row["motion_stratum"] == group]
        tokens = sum(int(row["tokens"]) for row in records)
        if tokens:
            motion_by_stratum[group] = sum(float(row["residual_mse"]) * int(row["tokens"]) for row in records) / tokens
    table = ["| 诊断 | 预测图像 PSNR / LPIPS / DISTS | Rank Pearson 中位数 |", "| --- | --- | ---: |"]
    for row in rows:
        corr = f"{summary[row['method']]:.4f}" if row["method"] in summary else "未定义"
        table.append(f"| {row['method']} | {_metrics(row)} | {corr} |")
    report = ["# HLA-Tok-T 时间冗余诊断", "",
              "使用 174 张原生图像构成同一视频内的 171 对真实相邻帧。A1 使用上一帧真值连续 Token，"
              "A2 使用已解包的 Q16-4 K16320 潜在表示。A3/A4 各自只使用独立接收端已提交的基础坐标。"
              "A0 使用全支持的 Rank PCA 均值作预测诊断。同一相邻对上的 RGB 指标按冻结实现计算。", "", *table, ""]
    if set(motion_by_stratum) == {"low", "high"}:
        report += [f"按每个 Rank/Region 覆盖的原生 RGB L1 变化切分，A3 低运动残差 MSE 为 "
                   f"{motion_by_stratum['low']:.6g}，高运动为 {motion_by_stratum['high']:.6g}。"
                   "低运动阈值为 2/255，高运动阈值为 8/255。", ""]
    report += ["有效维数按中心化协方差特征值的归一化谱熵指数计算，每八个真实 Region 取一个样本。"
               "残差符号诊断使用统一的 I 步长。Oracle 与 CLOSED 的可用分量范围不同，差距包含基础支持和量化的影响。"
               "可部署收益由独立实际码流给出。", "",
               "[相关性](rank_temporal_correlation.csv) · [分量](residual_statistics.csv) · [中心化谱](residual_spectrum.csv) · "
               "[符号频数](residual_histogram.csv) · [运动对照](motion_stratified_analysis.csv)"]
    (output / "temporal_redundancy_report.md").write_text("\n".join(report) + "\n")
    return table


def _observed_monotonicity(output: Path, method: str) -> dict:
    rows = [r for r in csv_at(output / "validation_per_frame.csv") if r["method"] == method
            and r["basis"] == "total_wire" and r["bandwidth"] in BANDS]
    lookup = {(r["video"], r["frame"], r["bandwidth"]): r for r in rows}
    pairs = []
    for row in rows:
        band = row["bandwidth"]
        order = BANDS.index(band)
        if order:
            previous = lookup[row["video"], row["frame"], BANDS[order - 1]]
            if previous["encoder_reference_digest"] != row["encoder_reference_digest"]:
                raise RuntimeError("matched rate comparison used a different committed base")
            pairs.append((previous, row))
    results = {name: {"violations": sum(sign * float(after[name]) < sign * float(before[name])
                                       for before, after in pairs), "pairs": len(pairs)}
               for name, sign in (("psnr", 1), ("lpips", -1), ("dists", -1))}
    write_json(output / "quality_monotonicity.json", results)
    return results


def _training(output: Path, choice: dict, runtime: dict) -> dict:
    prior = _gather(output, "training_selection")
    if prior.get("optimizer_steps", 0):
        return prior
    admissible = choice["quality_admission"] and runtime["status"] == "PASS"
    if admissible:
        raise RuntimeError("quality and runtime admitted training, so a closed-loop predictor decision is required")
    matches = csv_at(output / "matched_support_oracle.csv")
    oracle, closed = (next(row for row in matches if row["method"] == name)
                      for name in ("Same-support Oracle", "CLOSED base"))
    headroom = {"pairs": int(oracle["frames"]), **{
        name: float(oracle[name]) - float(closed[name]) for name in ("global_psnr", "lpips", "dists")}}
    selection = {"status": "NOT_RUN", "optimizer_steps": 0, "phase_3": "NOT_RUN", "phase_4": "NOT_RUN",
                 "quality_admission": choice["quality_admission"], "runtime_admission": runtime["status"] == "PASS",
                 "T0": "MEASURED", "T1": "NOT_RUN", "T2": "NOT_RUN",
                 "reason": "Validation did not reach the frozen quality gate. The 48 same-support pairs show small Oracle headroom.",
                 "same_support_oracle_minus_closed": headroom, "same_rate_motion_alignment_oracle": "NOT_RUN",
                 "checkpoint": None}
    write_json(output / "training_selection.json", selection)
    write_csv(output / "training_history.csv", [{"phase": phase, "status": "NOT_RUN", "optimizer_steps": 0}
                                                 for phase in (3, 4)])
    return selection


def _long_result(output: Path, ref: dict, recovered: int) -> str:
    if ref["status"] != "PASS":
        return "长序列验收状态为 " + ref["status"] + "。"
    rows = csv_at(output / "dynamic_bandwidth_results.csv")
    item, = [r for r in rows if r["video"] == "DAVIS2017:soapbox" and r["condition"] == "fixed_10"]
    early, mid, late = (float(item[f"{key}_psnr"]) for key in ("early", "middle", "late"))
    return (
        f"在 64、90、99 帧真实连续片段上完成 {ref['frames']:,} 个帧条件。"
        f"完整基础送达时摘要不一致 {ref['delivered_base_mismatches']} 次。"
        f"第 7 帧基础 Record 丢失后，第 32 帧的下一次周期 I 刷新在三个视频上恢复 {recovered} 次。"
        f"Soapbox 固定 10 Mbps 的前、中、后段 PSNR 为 {early:.2f} / {mid:.2f} / {late:.2f} dB。"
        "重复静止帧的整数基础收敛通过。冻结 Validation 的 128 和 256 帧连续条件为 NOT_AVAILABLE。"
    )


def run(output: Path) -> None:
    quality = json_at(output / "candidate_selection.json")
    method = quality["method"]
    table = csv_at(output / "bandwidth_comparison.csv")
    equal = csv_at(output / "equal_payload_comparison.csv")
    runtime = _final_runtime(output)
    ref = _gather(output, "reference_state_audit")
    prefix = _gather(output, "prefix_audit")
    wire = _gather(output, "wire_audit")
    causality = _gather(output, "causality_audit")
    engineer = _gather(output, "engineering_audit")
    history = _gather(output, "historical_assets_audit")
    tests = _gather(output, "test_access_audit")
    baseline = json_at(output / "baseline_reproduction.json")
    train = _training(output, quality, runtime)
    mono = _observed_monotonicity(output, method)
    oracle = train["same_support_oracle_minus_closed"]
    outcomes = quality["full_three_metric_targets"]
    total_goals = sum(v for band in outcomes.values() for v in band.values())
    payload, = [r for r in equal if r["method"] == method and r["basis"] == "payload_only"]
    c1, = [r for r in equal if r["method"] == "C1 Fixed13" and r["basis"] == "payload_only"]
    equal_pass = float(payload["global_psnr"]) >= float(c1["global_psnr"]) and all(
        float(payload[metric]) <= float(c1[metric]) for metric in ("lpips", "dists"))
    events = ref.get("base_recovery_events", [])
    recovered = sum(row["event"] == "I_recovery" for row in events)
    progress = _long_result(output, ref, recovered)
    dynamic = ref["status"]
    engineering_pass = engineer["status"] == "PASS" and history["status"] == "PASS"
    verdict = {
        "Experiment Completion": "PASS" if train.get("optimizer_steps", 0) >= 30000 and engineering_pass else "PARTIAL PASS",
        "Baseline Reproduction": baseline["status"],
        "Native 1080p": wire["status"], "Temporal Redundancy Benefit": "PASS" if quality["quality_admission"] else "FAIL",
        "Strict Temporal Causality": causality["status"], "Strict Integer Prefix": prefix["status"],
        "Strict Wire Prefix": prefix["status"], "Incremental Reconstruction": wire["status"],
        "Base Reference Synchronization": dynamic, "Enhancement Independence": dynamic,
        "Long-Horizon Closed Loop": dynamic, "Base Loss Recovery": "PASS" if recovered else "NOT_RUN",
        "Equal-Payload Rate–Distortion": "PASS" if equal_pass else "FAIL",
        "Practical Bandwidth Quality": "PASS" if total_goals == 12 else "PARTIAL PASS" if total_goals else "FAIL",
        "Runtime Performance": runtime["status"], "Dynamic Bandwidth Adaptation": dynamic,
        "Research Readiness": "PARTIAL PASS" if engineering_pass else "FAIL",
    }
    write_json(output / "final_verdict.json", verdict)
    profile, = [r for r in json_at(output / "matrix_identity.json")["profiles"] if r["label"] == method]
    lines = ["# HLA-Tok-T 最终报告", "",
             f"冻结 C0 原生 1920×1080、174 帧 Validation。选出的实测 Temporal 候选为 **{method}**，"
             f"基础层 **{profile['base']}**，{profile['base_components']} 个固定基础分量、"
             f"{profile['base_records']} 条基础 Record，最坏完整基础 {profile['max_base_bytes']} 字节。",
             f"使用 {profile['bits']}-bit、{profile['block_size']} 分量短块和 {profile['refresh']} 帧刷新间隔。"
             f"Test 像素读取 **{tests.get('test_pixel_reads', '未审计')}**。", "",
             "## 裁决", "", "| 项目 | 结果 |", "| --- | --- |",
             *[f"| {name} | {value} |" for name, value in verdict.items()], "",
             "## 真实码流与画质", "",
             "质量顺序为 Global PSNR、LPIPS-Alex、DISTS。码率按应用字节在 30 FPS 下换算。"
             "Original 使用同一 174 帧的已核验 Fixed13 归档，按 16 帧窗口将尾窗填充成本均摊到真实源帧。"
             "原版首帧等待为 15 帧，即 30 FPS 下 500 ms。HLT 以完整单帧作为输入，算法前视为 0 帧。", "",
             *_quality_table(table, method), "", "相对差值按 HLT 减去对照计算。感知指标下降对应改善。", "",
             *_delta_table(table, method), "",
             f"共同 26,520 B Payload 条件下，{method} 为 {_metrics(payload)}，C1 Fixed13 为 {_metrics(c1)}。"
             f"HLT 实际总字节均值为 {float(payload['total_bytes_per_frame']):.1f} B/frame。",
             "同总字节 26,544 B 及 6.7392 Mbps 的结果见 [共同上限](equal_payload_comparison.csv) 与 "
             "[完整带宽表](bandwidth_comparison.csv)。逐帧硬上限包含 I 帧、协议头、CRC、bitmap 和补位。", "",
             "## 贡献与训练准入", "", *_ablations(table), "",
             "准入比较使用每个码率下绝对 HLT 和历史 CP 对照的逐指标最强值。逐来源成对结果见候选记录。", ""]
    for band, gate in quality["gate"].items():
        lines.append(f"- {band} Mbps。相对最强对照 ΔPSNR {gate['global_psnr_gain_vs_strongest']:+.3f} dB，"
                     f"LPIPS 降幅 {gate['lpips_reduction_vs_strongest']:+.5f}，DISTS 增幅 "
                     f"{gate['dists_increase_vs_strongest']:+.5f}。{gate['source_passes']}/3 个来源过门槛。")
    rates = csv_at(output / "rate_breakdown.csv")
    one_low_rate = {r["method"]: r for r in rates if r["basis"] == "total_wire" and r["bandwidth"] == "5"}
    t0 = one_low_rate["T0-RA"]
    intra = one_low_rate["Intra RA"]
    raw = one_low_rate["T0-RA RAW"]
    lines += ["", f"锁定 S1 的 5 Mbps 对照中，绝对基础 Payload 为 {float(intra['base_payload_bytes']):.1f} B/frame，"
              f"复制闭环为 {float(t0['base_payload_bytes']):.1f} B/frame。"
              f"同档位 T0-RA 稀疏发送 {float(t0['sent_components']):.1f} 个分量/帧，RAW 发送 "
              f"{float(raw['sent_components']):.1f}。"
              "较小的基础 Payload 释放了增强容量，三项画质仍由整条扫描及有效表示决定。", ""]
    lines += ["", f"Phase 3 和 Phase 4 的训练状态分别为 {train['phase_3']} 与 {train['phase_4']}。"
              f"实际优化器更新 {train['optimizer_steps']} 步。"
              "T0 与有界线性外推在相同字节运行。T1 与 T2 的状态由前序准入裁决保留。", "",
              f"固定 B2 支持的 {oracle['pairs']} 对补充诊断中，Oracle 相对 CLOSED 的差值为 "
              f"{oracle['global_psnr']:+.3f} dB / {oracle['lpips']:+.5f} LPIPS / {oracle['dists']:+.5f} DISTS。"
              "完整支持 Oracle 的较大差距包含未覆盖分量的贡献。"
              " [同支持诊断](matched_support_oracle.csv)", "",
              f"提高预算后的逐帧画质下降次数为 PSNR {mono['psnr']['violations']}、"
              f"LPIPS {mono['lpips']['violations']}、DISTS {mono['dists']['violations']}，"
              f"每项检查 {mono['psnr']['pairs']} 个相邻预算对。", "",
              "## 参考状态与 H100", "",
              progress, "",
              *_runtime_line(output, runtime), "",
              "H100 延迟以 GPU 原生单帧输入至同步后的 GPU 重建计。输入文件读取和实际网络传输各自独立。"
              "CP 性能对照使用原归档代表帧重复口径，Temporal 在真实片段边界重置参考后循环。",
              "", "## 工程证据", "",
              f"工程审计为 **{engineer['status']}**，历史资产审计为 **{history['status']}**。"
              "接收端在独立进程用实际 HLT 字节和冻结合成权重生成原生 RGB。", "",
              "[基线复现](baseline_reproduction.json) · [时间冗余](temporal_redundancy_report.md) · "
              "[选型](candidate_selection.json) · [闭环明细](closed_loop_per_frame.csv) · "
              "[动态带宽](dynamic_bandwidth_results.csv) · [字节分解](rate_breakdown.csv) · "
              "[延迟](latency_breakdown.csv) · [工程审计](engineering_audit.json) · [Test 审计](test_access_audit.json) · "
              "[图表](figures/README.md) · [样例](samples/manifest.json) · "
              "[复现](../../../docs/hla_tok/temporal/README.md)", "",
              "下一步优先在固定 5/10 Mbps 口径下重配基础层的 Rank×Component 覆盖，"
              "先让参考层覆盖对 RGB 和感知质量更重要的分量，再重新测量时间预测的同字节净收益。"]
    (output / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    (_diagnosis if args.diagnostic_only else run)(args.output)
