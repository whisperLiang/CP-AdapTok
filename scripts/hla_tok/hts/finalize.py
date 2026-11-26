"""Archive the Phase 0 decision and produce the terminal report for a rejection."""

import argparse
import math
from pathlib import Path

import numpy as np

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.hts.native_comparison import strict_fit
from scripts.hla_tok.uf_v2.rate_analysis import at_quality, frontier
from scripts.hla_tok.v2.protocol import csv_at, json_at


def format_number(value, digits=5):
    return "区间外" if value is None else f"{float(value):.{digits}f}"


def native_table(points: list[dict]) -> list[str]:
    table = ["| 模型 | 配置 | 最低实测 Mbps | PSNR dB | LPIPS | 达到 25 dB 的最低 Mbps |",
             "|---|---|---:|---:|---:|---:|"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in points:
        row = item["summary"]
        groups.setdefault((row["model"], row["profile"]), []).append(row)
    for (variant, profile), rows in sorted(groups.items()):
        lowest = min(rows, key=lambda row: row["actual_mbps"])
        minimum = min((row["actual_mbps"] for row in rows if row["global_psnr"] >= 25), default=None)
        label = {"hts": "HT-S", "htl": "HT-L", "ld": "LD"}[variant]
        table.append(
            f"| {label} | {profile} | {lowest['actual_mbps']:.5f} | {lowest['global_psnr']:.3f} | "
            f"{lowest['lpips']:.5f} | {format_number(minimum)} |"
        )
    return table


def weighted_metrics(rows: list[dict]) -> dict:
    frames = sum(int(row["frames"]) for row in rows)
    return {"frames": frames, **{
        key: sum(int(row["frames"]) * float(row[key]) for row in rows) / frames
        for key in ("mse", "lpips", "dists")}}


def capacity_rows(native: Path) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in csv_at(native / "native_continuous_capacity.csv"):
        grouped.setdefault((int(row["qp"]), row["representation"]), []).append(row)
    return [capacity_gain(qp, weighted_metrics(grouped[(qp, "native")]),
                         weighted_metrics(grouped[(qp, "continuous_y")]))
            for qp in sorted({qp for qp, _ in grouped})]


def capacity_gain(qp: int, base: dict, continuous: dict) -> dict:
    return {"qp": qp, "valid_p_frames": base["frames"],
            "psnr_gain_db": 10 * math.log10(base["mse"] / continuous["mse"]),
            "lpips_reduction": base["lpips"] - continuous["lpips"],
            "dists_reduction": base["dists"] - continuous["dists"]}


def modeled_rows(points: list[dict], model: str, profile: str) -> list[dict]:
    return [point["summary"] for point in points if point["summary"]["model"] == model
            and point["summary"]["profile"] == profile]


def matching_rates(candidate: list[dict], reference: list[dict]) -> list[dict]:
    values = (("global_psnr", 26.0), ("lpips", 0.45), ("lpips", 0.28978))
    return [{"metric": metric, "quality": quality, "hts_mbps": at_quality(frontier(candidate, metric), metric, quality),
             "ld_mbps": at_quality(frontier(reference, metric), metric, quality)} for metric, quality in values]


def low_rate_observations(points: list[dict], old: dict) -> dict:
    native_hts = [point for point in points if point["summary"]["model"] == "hts"]
    minimum = min((point for point in native_hts if point["summary"]["global_psnr"] >= 25),
                  key=lambda point: point["summary"]["actual_mbps"])
    selected = minimum["summary"]
    profile = selected["profile"]
    official_ld = modeled_rows(points, "ld", profile)
    candidate = modeled_rows(points, "hts", profile)
    unit_ms = np.array([row["encoding_ms"] + row["decoding_ms"] for row in minimum["units"]], dtype=float)
    return {
        "minimum_hts_measured_at_25_db": selected, "matched_profile": profile,
        "hts_ld_equal_quality": {profile: matching_rates(modeled_rows(points, "hts", profile),
                                                          modeled_rows(points, "ld", profile))
                                 for profile in ("official", "matched")},
        "ld_at_selected_hts_quality": {
            metric: at_quality(frontier(official_ld, metric), metric, selected[metric])
            for metric in ("global_psnr", "lpips")},
        "hts_at_a4_lpips_mbps": at_quality(frontier(candidate, "lpips"), "lpips", old["lpips"]),
        "hts_at_a4_psnr_mbps": at_quality(frontier(candidate, "global_psnr"), "global_psnr", old["global_psnr"]),
        "hts_at_original_named_lpips_mbps": at_quality(frontier(candidate, "lpips"), "lpips", 0.28978),
        "native_codec_compute_fps": 1000 * selected["frames"] / unit_ms.sum().item(),
        "native_unit_compute_p95_ms": float(np.percentile(unit_ms, 95)),
        "strict_0486": strict_fit(minimum["units"], 0.486, 30),
    }


def matched_table(matches: dict) -> list[str]:
    rows = []
    for profile, cases in matches.items():
        for row in cases:
            name = "PSNR (dB)" if row["metric"] == "global_psnr" else "LPIPS-Alex"
            rows.append(f"| {profile} | {name} | {row['quality']:g} | {format_number(row['hts_mbps'])} | "
                        f"{format_number(row['ld_mbps'])} |")
    return rows


def render_report(verdict: dict, points: list[dict], observations: dict, capacities: list[dict]) -> str:
    selection = observations["minimum_hts_measured_at_25_db"]
    a4 = verdict["calibration_a4_base"]
    peaks = observations["strict_0486"]
    ld_rate = observations["ld_at_selected_hts_quality"]["lpips"]
    ld_sentence = ("该 LPIPS 超出相同配置的 LD 实测覆盖。" if ld_rate is None
                   else f"相同 LPIPS 下，LD 在实测区间内插值为 {ld_rate:.5f} Mbps。")
    lines = [
        "# HLA-Tok-HTS 原生审计裁决",
        "",
        "**Phase 0 = FAIL。** 单卡 H100 完成冻结 Calibration 的真实码流扫描。"
        "配置冻结所要求的低码率画质收益未出现，HT-S 集成和训练未获准入。"
        "正式 Validation 运行状态为 NOT_RUN，准入为 FAIL。Test 为 NOT_RUN，像素读取为 0。",
        "",
        "## 低码率实测",
        "",
        "四个原生 RGB 1920×1080 裁段共 128 帧。表中平均码率按 30 FPS 和完整物理字节计算。"
        "源清单的 FPS 为名义 30，实际采集时钟缺少独立测量。历史对齐采用 LD 32 帧和 HT 系列 33 帧 GOP。",
        "",
        *native_table(points),
        "",
        f"HT-S 最低达到 25 dB 的实测点为 **{selection['actual_mbps']:.5f} Mbps**、"
        f"{selection['global_psnr']:.3f} dB、LPIPS {selection['lpips']:.5f}。"
        f"配置为 {selection['profile']}、I QP {selection['qp_i']}、P QP {selection['qp_p']}。"
        + ld_sentence,
        "",
        f"V3 A4 Base-only 在相同 Calibration 样本上为 {a4['actual_mbps']:.5f} Mbps、"
        f"{a4['global_psnr']:.3f} dB、LPIPS {a4['lpips']:.5f}。"
        f"HT-S 达到该 LPIPS 时的实测区间内插值为 "
        f"{format_number(observations['hts_at_a4_lpips_mbps'])} Mbps，"
        f"达到相同 PSNR 时为 {format_number(observations['hts_at_a4_psnr_mbps'])} Mbps。",
        "",
        f"在上述 HT-S 端点，0.486 Mbps 给每个独立 I 帧 2,025 B。"
        f"最大 I 为 {selection['i_peak_bytes']:,} B，{peaks['i_units_over_budget']} 个 I 超限。"
        f"{peaks['affordable_units']}/{peaks['units']} 个编码单元单独满足物理预算。严格闭环需要完整 I 参考。",
        "",
        "## HT-S 与 LD 的等画质对照",
        "",
        "| 配置 | 指标 | 目标 | HT-S Mbps | LD Mbps |",
        "|---|---|---:|---:|---:|",
        *matched_table(observations["hts_ld_equal_quality"]),
        "",
        "## 原生容量与正确性",
        "",
        "| P QP | 有效 P 帧 | 不量化 y 的 PSNR 提升 dB | LPIPS 降低 | DISTS 降低 |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {row['qp']} | {row['valid_p_frames']} | {row['psnr_gain_db']:.3f} | "
                 f"{row['lpips_reduction']:.5f} | {row['dists_reduction']:.5f} |" for row in capacities)
    lines.extend([
        "",
        "连续诊断使用实际解码上下文与 `[1, 256, 68, 120]` 的 unquantized y。"
        "它展示官方表示的质量余量，连续路径的码率为空。",
        "",
        "独立 I 帧、1～7 帧尾块和 32 帧序列的官方 Base 字节、RGB、参考及上下文逐项相等。"
        "未来块扰动保持历史字节和显示一致，官方权重和 buffers 的哈希相等。",
        "",
        "## 研究与工程结论",
        "",
        "| 问题 | 当前证据 |",
        "|---|---|",
        "| 1. 官方 HT-S 相对 LD 的低码率压缩 | 两套配置在 0.486 Mbps 以下的实测重叠区间均无等画质省字节端点。 |",
        "| 2. HLA-Tok-HTS 原生 Base 保留 | 原生审计路径 parity 为 PASS。HLA-Tok-HTS 集成为 NOT_RUN。 |",
        "| 3. 最低可接受码率 | 见上表，以 25 dB 为本次 Calibration 的报告线。 |",
        "| 4. HLA-Tok-HTS 相对 V3 的等 LPIPS 码率 | NOT_RUN。原生对照的区间内插值见上文。 |",
        "| 5. 渐进增强的额外率失真收益 | NOT_RUN。 |",
        "| 6. 每增加 1 Mbps 的增强画质收益 | NOT_RUN。 |",
        "| 7. 块级因果、前缀、增量解码 | 原生块级因果为 PASS。增强前缀和增量解码为 NOT_RUN。 |",
        f"| 8. 原生 1080p 编解码 | 选定端点原生编解码计算合计 "
        f"{observations['native_codec_compute_fps']:.2f} FPS，单元 P95 "
        f"{observations['native_unit_compute_p95_ms']:.2f} ms。完整渐进运行时为 NOT_RUN。 |",
        "| 9. 含采集与网络的实际延迟 | NOT_RUN。 |",
        "| 10. 最终论文系统 | 当前实测支持将 HT-S 作为原生对照并报告负结果。 |",
        "",
        "Original AdapTok 的实测码率使用独立 16 帧 F13W 窗口的有效帧平均。"
        "此前 174 帧历史 Validation 的 1.39471 Mbps 与 0.28978 LPIPS 锚点保留其原始统计范围。"
        "本轮 Calibration 曲线使用相同 128 帧，并在各自实测质量覆盖内插值。",
        "",
        "## 归档",
        "",
        f"QP 与 I/P 组合共 {verdict['native_cases_completed']} 个。逐单元账本经过原生 SPS/IP 复解析，"
        f"归档 {verdict['byte_accounting']['streams']:,} 个完整原生码流。",
        "",
        "- [原生扫点](native_audit/qp_sweep.csv)",
        "- [低码率重叠比较](native_audit/native_comparison.json)",
        "- [逐单元真实字节账本](native_audit/byte_ledger.jsonl)",
        "- [旧资产与构建 SHA256](experiment_manifest.json)",
        "- [图件清单](figures/manifest.json)",
        "- [验证摘要](qa/summary.json)",
        "- [执行命令和退出码](logs/command_runs.jsonl)",
        "- [机器裁决](final_verdict.json)",
    ])
    return "\n".join(lines) + "\n"


def pixel_audit(native: Path) -> dict:
    originals = {
        "native_sweep": json_at(native / "pixel_reads.json"),
        "native_parity": json_at(native / "native_correctness.json")["pixels_read"],
        "continuous_native": json_at(native / "native_continuous_scope.json")["pixels_read"],
        "original_adaptok": json_at(native / "legacy/original/pixel_reads.json"),
        "v2_e1": json_at(native / "legacy/a2/summary.json")["read_counts"],
        "v3_a4": json_at(native / "legacy/a4/summary.json")["read_counts"],
    }
    validation_reads = sum(counts["validation"] for counts in originals.values())
    test_reads = sum(counts["test"] for counts in originals.values())
    if validation_reads or test_reads:
        raise ValueError("Phase 0 accessed a sealed evaluation split")
    return {"by_operation": originals, "validation": validation_reads, "test": test_reads}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    native = args.results / "native_audit"
    verdict = json_at(native / "phase_verdict.json")
    if verdict["verdict"] != "FAIL":
        raise ValueError("a terminal Phase 0 rejection needs a measured FAIL verdict")
    points = [json_at(path) for path in sorted(native.glob("*/*/point.json"))]
    observations = low_rate_observations(points, verdict["calibration_a4_base"])
    capacities = capacity_rows(native)
    counts = pixel_audit(native)
    result = {
        "verdict": "FAIL", "phases": {"0": "FAIL", **dict.fromkeys(map(str, range(1, 6)), "NOT_RUN")},
        "validation": "FAIL", "validation_execution": "NOT_RUN", "validation_pixel_read_count": counts["validation"],
        "test": "NOT_RUN", "test_pixel_read_count": counts["test"], "optimizer_steps": {"1": 0, "2": 0},
        "reason": "The measured native HT-S low-rate admission gate failed.",
        "native_observations": observations, "continuous_native_diagnostics": capacities,
    }
    write_json(args.results / "pixel_audit.json", counts)
    write_json(args.results / "final_verdict.json", result)
    (args.results / "final_report.md").write_text(render_report(verdict, points, observations, capacities))


if __name__ == "__main__":
    main()
