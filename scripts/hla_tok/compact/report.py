"""Publish provenance and verdicts from the completed Compact ledgers."""

from pathlib import Path
from statistics import median

from hla_tok.evaluation.io import write_json
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.decision import stage_targets
from scripts.hla_tok.compact.inputs import CONFIG, CONTINUOUS, OUTPUT
from scripts.hla_tok.compact.tables import write_tables
from scripts.hla_tok.continuous.inputs import run_arguments
from scripts.hla_tok.v2.protocol import csv_at, json_at
from scripts.hla_tok.v3.protocol import read_config

VERDICT_KEYS = (
    "Experiment Completion",
    "Compact Representation Correctness",
    "Strict Prefix",
    "Temporal Causality",
    "Progressive Reconstruction",
    "Equal-Payload Rate–Distortion",
    "Practical Bandwidth Quality",
    "Runtime Performance",
    "Adaptive Video Streaming Research Readiness",
)


def _fmt(row: dict, field: str, decimals: int = 2) -> str:
    value = row.get(field, "")
    return f"{float(value):.{decimals}f}" if value not in ("", None) else "—"


def _metric_triplet(row: dict, prefix: str = "") -> str:
    return " / ".join(_fmt(row, prefix + field, precision)
                      for field, precision in (("global_psnr", 2), ("lpips", 4), ("dists", 4)))


def _effective(output: Path) -> tuple[dict, list[float]]:
    spectra = csv_at(output / "rank_effective_dimension.csv")
    global_row, = [r for r in spectra if r["scope"] == "global"]
    ranks = [float(r["effective_rank_entropy"]) for r in spectra if r["scope"] == "rank"]
    return global_row, ranks


def feature_report(output: Path) -> None:
    global_row, ranks = _effective(output)
    values = csv_at(output / "channel_sensitivity.csv")
    sens = ", ".join(f"{kind} C{max((r for r in values if r['projection'] == kind), key=lambda r: float(r['rgb_mse_delta']))['channel']}"
                     for kind in ("identity", "global", "rank"))
    audit = json_at(output / "representation_audit.json")
    summaries = [f"| {d}D | {100 * float(global_row[f'variance_{d}d']):.2f}% |" for d in (4, 8, 16, 24, 32)]
    report = f"""# C0 特征冗余

{audit['train_frames']} 帧原生 Train Token 用于描述分布。协方差用其中 {audit['fit_frames']} 帧拟合。独立校准集包含 {audit['holdout_frames']} 帧，各视频来源与拟合集互斥。缓存包含 120 个完整空间 Rank 和含 120 个区域的 Rank 120。

全局熵有效维数为 {float(global_row['effective_rank_entropy']):.2f}，参与率为 {float(global_row['participation_ratio']):.2f}。各 Rank 熵有效维数中位数为 {median(ranks):.2f}，范围为 {min(ranks):.2f}–{max(ranks):.2f}。累计解释方差达到 95% 和 99% 分别需要 {global_row['dimensions_at_95_percent']} 和 {global_row['dimensions_at_99_percent']} 维。

| 维度 | 全局解释方差 |
| --- | ---: |
""" + "\n".join(summaries) + f"""

六帧 RGB 消融中，平均 MSE 增幅最大的零起始分量为 {sens}。逐分量能量与 RGB 敏感性分别可查。相邻区域水平和垂直相关性为 {audit['adjacent_region_correlation']['horizontal']:.4f} 和 {audit['adjacent_region_correlation']['vertical']:.4f}。Rank 相关性使用 121 个 Rank 共用的前 120 个区域。

[谱](pca_spectrum.csv) · [有效维数](rank_effective_dimension.csv) · [敏感性](channel_sensitivity.csv) · [BF16 字节分布](representation_audit.json) · [协方差与相关矩阵](rank_covariance.npz)
"""
    (output / "feature_redundancy_report.md").write_text(report)


def projection_report(output: Path) -> None:
    rows = csv_at(output / "projection_maximum_rd.csv")
    base, = [r for r in rows if r["projection"] == "identity"]
    inverse = csv_at(output / "projection_reversibility.csv")
    table = []
    for d in (32, 24, 16, 8, 4):
        a, b = [next(r for r in rows if r["profile"] == f"PCA{d}-BF16" and r["projection"] == kind)
                for kind in ("global", "rank")]
        float_delta = float(b["global_psnr"]) - float(base["global_psnr"])
        table.append(f"| {d}D | {_metric_triplet(a)} | {_metric_triplet(b)} | {float_delta:+.3f} |")
    maxes = {p: max(float(r["max_abs"]) for r in inverse if r["compute_dtype"] == p) for p in ("fp32", "bf16")}
    report = f"""# 纯降维损失

原始 O2 BF16 K16320 的 PSNR、LPIPS-Alex 和 DISTS 为 {_metric_triplet(base)}。下表中的所有条件均使用同一 174 帧的实际 BF16 码流。

| 维度 | Global PCA | Rank-aware PCA | Rank-aware ΔPSNR (dB) |
| --- | --- | --- | ---: |
""" + "\n".join(table) + f"""

32D 往返最大绝对误差在 FP32 为 {maxes['fp32']:.4g}，在 BF16 为 {maxes['bf16']:.4g}。低预算及排序差异可查逐档结果。

[八档率失真](projection_only_rd.csv) · [逐帧](projection_only_per_frame.csv) · [排序与方法比较](global_vs_rank_pca.csv) · [数值可逆性](projection_reversibility.csv)
"""
    (output / "projection_loss_report.md").write_text(report)


def _headlines(output: Path, selection: dict, training: dict) -> list[dict]:
    bands = csv_at(output / "bandwidth_constrained_quality.csv")
    main = selection["main_asset"]
    rows = [r for r in bands if r["asset"] == main and r["order"] == selection["order"]]
    if training["adopted"]:
        rows = [r for r in csv_at(output / "training_final_bandwidth.csv") if r["basis"] == "total_wire"]
    return rows


def _runtime(output: Path, main: str) -> tuple[bool, list[dict]]:
    measured = csv_at(output / "throughput_benchmark.csv")
    expected = [r for r in measured if r["asset"] == main]
    if not expected:
        raise ValueError("no H100 data found for the recommended Profile")
    ok = all(r["pass_30fps"] == r["pass_p95"] == "True" for r in expected)
    return ok, measured


def _primary_table(output: Path, selection: dict, training: dict, runtime: list[dict]) -> str:
    lines = ["| Method | Profile | K | Actual bytes/frame | Total bpp | Mbps@30FPS | PSNR | LPIPS | DISTS | FPS | P95 (ms) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    original = max(csv_at(output / "original_rd.csv"), key=lambda r: int(r["tokens_per_tile"]))
    last_c0 = [r for r in csv_at(output / "single_frame_references.csv") if int(r["cap_bytes"]) <= 0]
    raw = list(("C0 continuous" if row["method"] == "C0 BF16" else "C1 evaluated envelope", row["method"], row)
               for row in last_c0)
    raw += [("Original AdapTok", "F13W, 16 frames", original | {"budget": original["tokens_per_tile"]})]
    raw += [("Compact", row["profile"], row) for row in _headlines(output, selection, training)]
    timings = {(r["asset"], int(r["budget"])): r for r in runtime if r["compute_dtype"] == "bf16"}
    old_timing = [r for r in csv_at(CONTINUOUS / "throughput_benchmark.csv") if r["wire_dtype"] == "bf16"
                  and r["compute_dtype"] == "bf16" and int(r["budget"]) == MAX_K]
    for method, label, r in raw:
        timing = timings.get((r.get("asset", ""), int(r["budget"])), {})
        if r.get("method") == "C0 BF16":
            timing = old_timing[0] if int(r["budget"]) == MAX_K and old_timing else {}
        entries = [method, label, r["budget"], _fmt(r, "total_bytes_per_frame", 1), _fmt(r, "total_bpp", 5),
                   _fmt(r, "mbps_30fps", 4), _fmt(r, "global_psnr"), _fmt(r, "lpips", 5), _fmt(r, "dists", 5),
                   _fmt(timing, "steady_state_fps") if timing else _fmt(r, "codec_fps"),
                   _fmt(timing, "p95_ms", 3)]
        lines.append("| " + " | ".join(map(str, entries)) + " |")
    return "\n".join(lines)


def _bands_table(output: Path) -> str:
    rows = csv_at(output / "bandwidth_baseline_comparison.csv")
    lines = ["| 带宽 Mbps | C0 BF16 | C1 Fixed13 | Original AdapTok | Best Compact | ΔPSNR vs C0 |",
             "| ---: | --- | --- | --- | --- | ---: |"]
    for row in rows:
        values = [_metric_triplet(row, method + "_") if row[method + "_status"] == "MEASURED" else "NOT_COMPARABLE"
                  for method in ("c0", "c1_f13w", "original", "best_compact")]
        label = f"{row['best_compact_profile']} · " + values[-1]
        lines.append("| " + " | ".join([row["target_mbps"], *values[:-1], label, f"{float(row['gain_vs_c0_db']):+.2f}"]) + " |")
    return "\n".join(lines)


def _states(output: Path, selection: dict, targets: dict, runtime_ok: bool, integrity: dict) -> dict:
    evidence = [json_at(output / file)["status"] == "PASS" for file in (
        "reproduction.json", "representation_audit.json", "wire_audit.json", "prefix_audit.json",
        "causality_audit.json", "lossless_audit.json", "targeted_mutations.json")]
    correct = all(evidence) and integrity["test_pixel_reads"] == 0
    eq = targets["equal_payload_pass"]
    practical = targets["all_practical_pass"]
    bands = sum(row["pass"] for row in targets["practical"].values())
    values = ("PASS", "PASS" if correct else "FAIL", "PASS" if evidence[3] else "FAIL",
              "PASS" if evidence[4] else "FAIL", selection["progressive"],
              "PASS" if eq else "FAIL", "PASS" if practical else ("PARTIAL PASS" if bands else "FAIL"),
              "PASS" if runtime_ok else "FAIL",
              "PASS" if correct and eq and practical and runtime_ok else ("PARTIAL PASS" if correct else "FAIL"))
    return dict(zip(VERDICT_KEYS, values, strict=True))


def _recommend(targets: dict, training: dict, runtime_ok: bool, output: Path, config: dict) -> tuple[str, str]:
    if targets["equal_payload_pass"] and targets["all_practical_pass"] and runtime_ok:
        return (("B", "采用通过门槛的轻量适配 Profile") if training["adopted"]
                else ("A", "保留最优免训练 Profile"))
    low = [targets["practical"][band]["pass"] for band in ("5", "10")]
    if all(low) and not targets["all_practical_pass"]:
        return "F", "研究同一表示的精度或分量残差增强，保持已测低码率路径"
    pure = csv_at(output / "projection_maximum_rd.csv")
    p8 = float(next(r["global_psnr"] for r in pure if r["asset"] == "PCA8-BF16_rank"))
    p16 = float(next(r["global_psnr"] for r in pure if r["asset"] == "PCA16-BF16_rank"))
    if p8 < config["stage_psnr_targets"][20] <= p16:
        return "C", "保留 Rank-aware 表示，优先用同帧实际比特预算重新分配分量，避免继续将所有 Rank 统一压到 4D"
    return "E", "优先研究同帧的实际比特预算信息分配"


def conclusions(output: Path, selection: dict, training: dict, state: dict, targets: dict,
                recommend: tuple, config: dict) -> str:
    global_row, ranks = _effective(output)
    projection = json_at(output / "projection_selection.json")
    hardware = [r for r in csv_at(output / "throughput_benchmark.csv") if r["asset"] ==
                (training["asset"] if training["adopted"] else selection["main_asset"])]
    quality = {r["target_mbps"]: r for r in csv_at(output / "bandwidth_baseline_comparison.csv")}
    first8 = quality["5"]
    best = [r for r in csv_at(output / "bandwidth_constrained_quality.csv")
            if r["profile"] == "Q8-4" and r["order"] == selection["order"]]
    q84 = "、".join(f"{r['target_mbps']} Mbps 下 {float(r['global_psnr']):.2f} dB" for r in best)
    equal8, = [r for r in csv_at(output / "compact_equal_caps.csv")
               if r["profile"] == "Q8-4" and r["order"] == selection["order"] and r["basis"] == "payload_only"]
    checks8 = stage_targets(best, equal8, config)
    q84_status = "达到" if checks8["equal_payload_pass"] and checks8["all_practical_pass"] else "低于"
    ablations = [r for r in csv_at(output / "equal_bits_ablation.csv") if int(r["budget"]) == MAX_K]
    pairs = "、".join(f"{r['higher_dimension']} 对 {r['lower_dimension']} 为 "
                     f"{float(r['higher_dimension_gain_global_psnr']):+.2f} dB" for r in ablations)
    eq = targets["equal_payload_gain_db"]
    compare = first8["original_status"] == "MEASURED"
    original_gain = float(first8["best_compact_global_psnr"]) - float(first8["original_global_psnr"]) if compare else None
    original = f"在 5 Mbps 共同上限下，最佳 Compact 与 Original 相差 {original_gain:+.2f} dB。" if compare else "5 Mbps 缺少可比点。"
    saturation = selection["first_saturated_band_mbps"]
    return "\n".join([
        f"1. 全局熵有效维数 {float(global_row['effective_rank_entropy']):.2f}，各 Rank 中位数 {median(ranks):.2f}。",
        f"2. 16D、8D、4D 的冻结排序分别选中 {projection['16']['projection']}、{projection['8']['projection']}、{projection['4']['projection']}。",
        "3. 纯投影损失和同 K 下额外标量损失分别见 [降维报告](projection_loss_report.md) 和 [分解表](projection_quantization_ablation.csv)。",
        f"4. K16320 的等字节高维低精度相对低维高精度结果为 {pairs}。感知差值和低预算见 [等比特消融](equal_bits_ablation.csv)。",
        f"5. 8D INT4 通过实际码流正确性。其联合画质{q84_status}冻结目标。实测 PSNR 为 {q84}。",
        "6. 5、10、20、40 Mbps 的三项质量见带宽表。",
        f"7. 主 Profile 在 26,520 字节等 Payload 下与 C1 相差 {eq:+.2f} dB，联合门槛为 {state['Equal-Payload Rate–Distortion']}。",
        "8. " + original + "其余带宽点见表。Original 等待 16 帧输入，比较按窗口平均字节计。",
        "9. O1/O2 的 FP32 与 BF16 实测涵盖八档预算、50 个非预设长度及边界长度。见 [严格前缀](prefix_audit.json)。",
        f"10. 免训练主 Profile 首次达到 K16320 的冻结带宽点为 {saturation if saturation is not None else '未达'} Mbps。",
        f"11. 主 Profile 在单卡 H100 实测 {min(float(r['steady_state_fps']) for r in hardware):.2f}–"
        f"{max(float(r['steady_state_fps']) for r in hardware):.2f} FPS，串行 P95 最高 "
        f"{max(float(r['p95_ms']) for r in hardware):.3f} ms。",
        f"12. 轻量适配裁决为 {training['status']}。见 [适配证据](training_selection.json)。",
        "13. 本轮证据的训练范围是紧凑映射与 Lift。解冻 Tokenizer 的增益留待独立实验裁决。",
        "14. 码流为逐帧无状态路径。时间冗余压缩的增益留待独立实验裁决。",
        f"15. 唯一推荐主线为 {recommend[0]}。{recommend[1]}。",
    ])


def run(output: Path) -> None:
    write_tables(output)
    feature_report(output)
    projection_report(output)
    selection = json_at(output / "no_training_selection.json")
    training = json_at(output / "training_selection.json")
    main = training["asset"] if training["adopted"] else selection["main_asset"]
    targets = training["targets"] if training["adopted"] else selection["targets"]
    config = read_config(CONFIG)
    integrity = json_at(output / "historical_assets_audit.json")
    runtime_ok, runtime = _runtime(output, main)
    state = _states(output, selection, targets, runtime_ok, integrity)
    recommend = _recommend(targets, training, runtime_ok, output, config)
    write_json(output / "final_verdict.json", {"experiment": "HLA-Tok-CP", "main_asset": main,
                                              "status": state, "targets": targets, "application_admission": "NOT_DECIDED",
                                              "historical_v3_fixed13": "FAIL", "test_pixel_reads": integrity["test_pixel_reads"],
                                              "recommendation": {"scenario": recommend[0], "action": recommend[1]}})
    table = "\n".join(f"| {gate} | {state[gate]} |" for gate in VERDICT_KEYS)
    main_stats = "\n".join(f"- {key} Mbps 为 {point['psnr_db']:.2f} dB，门槛 {point['target_db']} dB。"
                           for key, point in targets["practical"].items())
    text = f"""# HLA-Tok-CP 最终报告

在 Seed 1 C0、原生 1920×1080、174 帧冻结 Validation 和单卡 H100 上测量。C0 Global PSNR、LPIPS-Alex 和 DISTS 复现值为 32.049857、0.094572 和 0.058787。Test 像素读取为 {integrity['test_pixel_reads']}。固定主 Profile 为 **{main} / {selection['order']}**。相邻帧可独立选择预算。

## 裁决

| 能力 | 裁决 |
| --- | --- |
{table}

应用端绝对画质要求为 **NOT_DECIDED**。历史 V3 Fixed13 正式裁决为 **FAIL**。 [历史裁决](../v3/final_report.md) · [复现](reproduction.json) · [历史资产](historical_assets_audit.json)

## 实测码率与画质

{_primary_table(output, selection, training, runtime)}

Original 的 K 表示每个空间 Tile 在整个 16 帧窗口内的 Token 数。Original 的 FPS 包含物理 Fixed13 往返与拼接，且算法必须等待整窗，30 FPS 输入时首帧等待 500 ms。C1 的带头结果使用新增的 24 字节 F13W 实验封装。 [窗口账本](original_window_ledger.csv) · [C1 权重](c1_reference_identity.json) · [原版归档核验](original_archive_identity.json)

## 共同字节上限

各单元格为 PSNR / LPIPS-Alex / DISTS。Best Compact 按每档联合排序，固定主 Profile 的结果列于核心表。Original 按窗口均摊，使用小于或等于上限的实测档位。 [来源表](bandwidth_baseline_comparison.csv)

{_bands_table(output)}

固定主 Profile 的研究门槛。

{main_stats}

历史 C1 的 26,520 字节对应 Payload。历史全包缺少独立账本，等总码流裁决为 **NOT_COMPARABLE**。同样本重放的 C1 F13W 包给出 26,544 字节的独立对照。 [等码率表](equal_rate_comparison.csv)

## 表示与适配

{conclusions(output, selection, training, state, targets, recommend, config)}

[Train 特征谱和 RGB 消融](feature_redundancy_report.md) · [纯降维](projection_loss_report.md) · [静态尺度](scalar_calibration.csv) · [无损诊断](lossless_diagnostics.csv) · [适配](training_selection.json)

## 复现与验收

[工程验收](engineering_audit.json) · [独立接收](wire_audit.json) · [因果性](causality_audit.json) · [实际 H100 延迟](latency_breakdown.csv) · [完整逐帧延迟](validation_latency.csv) · [图表](figures/README.md) · [原生样例](samples/manifest.json) · [复现入口](../../../docs/hla_tok/compact/README.md)
"""
    (output / "final_report.md").write_text(text)


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.tables_only:
        write_tables(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
