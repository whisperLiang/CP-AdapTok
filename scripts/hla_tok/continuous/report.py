"""Build decision tables from completed native, wire and H100 artifacts."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.continuous.inputs import METRICS, RESULTS
from scripts.hla_tok.v2.protocol import csv_at, json_at

RUNS = ("reproduction", "native_run", "ordered_run", "dynamic_run", "wire-checks_run", "wire_run",
        "benchmark_run", "samples_run")


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def _quality(row) -> list[str]:
    return [_fmt(row["global_psnr"], 2), _fmt(row["lpips"], 5), _fmt(row["dists"], 5)]


def _write(output: Path, name: str, *blocks: str) -> None:
    (output / f"{name}.md").write_text("\n\n".join(blocks) + "\n")


def _progress_text(progressive: dict) -> str:
    records = []
    for order, data in progressive.items():
        max_violation = {metric: max(pair["frame_violation_fraction"][metric] for pair in data["adjacent"])
                         for metric in METRICS}
        records.append([order, data["status"], *[f"{100 * max_violation[metric]:.1f}%" for metric in METRICS]])
    return _table(("顺序", "汇总渐进门槛", "逐帧 PSNR 最大违反率", "逐帧 LPIPS 最大违反率",
                   "逐帧 DISTS 最大违反率"), records)


def _ordering(output: Path, progressive: dict, decision: dict) -> None:
    raw = csv_at(output / "fixed_budget.csv") + csv_at(output / "ordered_fixed_budget.csv")
    results = {(row["order"], int(row["budget"])): row for row in raw}
    adjacent = [[order, f'{pair["lower"]} → {pair["upper"]}', *[_fmt(pair["gain"][m], 5) for m in METRICS],
                 "PASS" if pair["pass"] else "FAIL"]
                for order, data in progressive.items() for pair in data["adjacent"]]
    comparison = _table(("K", "O0 PSNR / LPIPS / DISTS", "O1 PSNR / LPIPS / DISTS",
                         "O2 PSNR / LPIPS / DISTS"), [
        [budget, *[" / ".join(_quality(results[order, budget])) for order in ("O0", "O1", "O2")]]
        for budget in sorted({k for _, k in results})
    ])
    _write(output, "ordering_comparison", "# 静态排序与八档重建",
           f'冻结低档选择规则得到主排序 **{decision["primary_order"]}**。'
           "O0、O1、O2 在 K16320 的完整向量和 RGB 输出均恢复原始 C0。",
           comparison, _progress_text(progressive),
           "下表三项增益均以正值表示改善。逐帧和各来源的邻档违反率保留在结构化记录。"
           " [渐进质量记录](progressive_quality.json)",
           _table(("顺序", "邻档 K", "PSNR 增益", "LPIPS 改善", "DISTS 改善", "容差"), adjacent))


def _frame_precision_deltas(output: Path) -> str:
    def key(row):
        return tuple(row[field] for field in ("order", "budget", "video", "frame"))

    sources = ("fixed_budget_per_frame.csv", "ordered_fixed_budget_per_frame.csv")
    raw = {key(row): row for source in sources for row in csv_at(output / source)}
    frames = csv_at(output / "wire_rate_distortion_per_frame.csv")
    fields = ("mse", "psnr", "lpips", "dists")
    peak = {field: max(abs(float(row[field]) - float(raw[key(row)][field])) for row in frames)
            for field in fields}
    display = "、".join(f"{field} {_fmt(delta, 10)}" for field, delta in peak.items())
    return f"逐帧共配对 {len(frames):,} 条实际观测，指标最大绝对差为 {display}。"


def _wire_comparison(output: Path, wire: list[dict]) -> None:
    rows = []
    for label in ("fp32", "fp16", "bf16"):
        curves = [row for row in wire if row["wire_dtype"] == label]
        endpoint = next(row for row in curves if int(row["budget"]) == MAX_K)
        max_delta = [max(abs(float(row[f"wire_minus_raw_{metric}"])) for row in curves) for metric in METRICS]
        rows.append([label.upper(), int(float(endpoint["payload_bytes_per_frame"])),
                     int(float(endpoint["total_bytes_per_frame"])), _fmt(endpoint["total_bpp"], 8),
                     *[_fmt(endpoint[f"mbps_{fps}fps"], 5) for fps in (24, 30, 60)],
                     *[_fmt(delta, 8) for delta in max_delta]])
    _write(output, "wire_format_comparison", "# 实际浮点字节与格式损失",
           "质量矩阵以 BF16 计算执行，包含三种顺序、八档和三种 Wire 类型。"
           "每条观测经过 GPU 到 CPU、真实字节、解析以及 CPU 到 GPU。"
           "下表绝对质量差采用同顺序、同预算的未序列化重建为参照，汇总范围覆盖全部正式档。",
           _table(("Wire", "K16320 Payload B/帧", "K16320 Total B/帧", "Total bpp", "Mbps 24 FPS",
                   "Mbps 30 FPS", "Mbps 60 FPS", "最大绝对 ΔPSNR", "最大绝对 ΔLPIPS", "最大绝对 ΔDISTS"), rows),
           _frame_precision_deltas(output) + " [完整率失真表](wire_rate_distortion.csv)"
           " · [逐帧字节与指标](wire_rate_distortion_per_frame.csv)")


def _equal_analysis(output: Path, comparison: list[dict]) -> None:
    equal = [row for row in comparison if row["comparison"] == "equal_payload_cap" and row["comparability"] == "PASS"]
    records = [[row["method"], row["order"] or "—", row["wire_dtype"].upper(), row["budget"],
                int(float(row["payload_bytes_per_frame"])), _fmt(row["payload_bpp"], 8), *_quality(row)]
               for row in equal]
    _write(output, "equal_rate_analysis", "# C1 Fixed13 等 Payload 帧预算",
           "C0 与 C1 使用相同的 174 帧原生 Validation、相同指标权重和逐帧无状态评价。"
           "C1 的 26,520 字节上限内，C0 FP32 可发送 K207，FP16/BF16 可发送 K414。"
           "两者实际 Payload 均为 26,496 字节，留下 24 字节不足容纳一个向量。"
           "连续包计入包头后为 26,524 字节。",
           _table(("路径", "排序", "Wire", "K", "Payload B/帧", "Payload bpp", "PSNR (dB)", "LPIPS", "DISTS"),
                  records),
           "C1 的完整包头账本缺失，等总码流为 **NOT_COMPARABLE**。"
           "K16320 等 Token 对比随同等 Payload 观测保存在原始表。 [完整对照](equal_rate_comparison.csv)",
           "原版 AdapTok 同样评价 174 帧原生 RGB，使用已归档感知权重。它使用 16 帧输入块，"
           "且公开记录的速率为 Fixed13 Payload。按本轮统一逐帧时间条件，公平等码率比较为 "
           "**NOT_COMPARABLE**。图 2 保留带时间口径标记的归档参考。 [归档参考](original_adaptok_context.csv)")


def _geometry_range(rows, key, digits=2):
    values = [float(row[key]) for row in rows]
    return f"{min(values):.{digits}f}–{max(values):.{digits}f}"


def _perf_report(output: Path, all_perf: list[dict]) -> dict:
    groups = []
    for path in ("float_wire_serial", "gpu_only"):
        for precision in ("fp32", "bf16"):
            pool = [row for row in all_perf if row["path"] == path and row["compute_dtype"] == precision]
            passes = [sum(row[key] == "True" for row in pool) for key in
                      ("minimum_30fps_pass", "ideal_60fps_pass", "p95_deadline_pass")]
            groups.append([path, precision.upper(), len(pool), _geometry_range(pool, "steady_state_fps"),
                           _geometry_range(pool, "p95_ms", 3), *[f"{value}/{len(pool)}" for value in passes]])
    wire = [row for row in all_perf if row["path"] == "float_wire_serial"]
    pass_all = sum(row["minimum_30fps_pass"] == row["p95_deadline_pass"] == "True" for row in wire)
    state = "PASS" if pass_all == len(wire) else "PARTIAL PASS" if pass_all else "FAIL"
    _write(output, "performance_report", "# 单卡 H100 物理码流性能",
           "Batch 1，原生 1920×1080。每配置 3 次重复，每次预热 100 帧并测量 1000 帧。"
           "在冻结 Validation 三个来源的九张代表帧间循环，采样次数是实际执行次数。"
           "同一 H100 上单独计时真实浮点封包和纯 GPU 路径。",
           _table(("路径", "计算", "配置数", "实际稳定 FPS 范围", "P95 ms 范围", "≥30 FPS", "≥60 FPS", "P95 ≤ 33.33 ms"),
                  groups),
           "模型权重和输入 RGB 已在 GPU 上。GPU 计算段用 CUDA Event，两个 Token 复制及字节编解包用 "
           "wall clock，完整串行延迟包含同步。CPU 内存项为进程 RSS 高水位。"
           "物理序列化不含磁盘 I/O 与外部网络等待。",
           "公开接口所有预算均运行完整 K16320 Encoder 的四个查询块。Recovery 使用收到的实际 Rank 数，"
           "RGB Decoder 每帧恢复原生像素。"
           " [阶段延迟](latency_breakdown.csv) · [每配置 FPS 和阈值](throughput_benchmark.csv)"
           " · [内存和参数](memory_benchmark.csv)")
    return {"status": state, "wire_configs": len(wire), "minimum_and_deadline_pass": pass_all,
            "fps_range": _geometry_range(wire, "steady_state_fps"), "p95_ms_range": _geometry_range(wire, "p95_ms", 3)}


def _core_table(wire: list[dict], perf: list[dict], primary: str) -> str:
    matches = {(row["wire_dtype"], row["budget"]): row for row in perf
               if row["path"] == "float_wire_serial" and row["compute_dtype"] == "bf16"}
    view = sorted((row for row in wire if row["order"] == primary), key=lambda row: (int(row["budget"]), row["wire_dtype"]))
    rows = []
    for row in view:
        run = matches[row["wire_dtype"], row["budget"]]
        rows.append([row["budget"], row["wire_dtype"].upper(), int(float(row["total_bytes_per_frame"])),
                     _fmt(row["total_bpp"], 8), _fmt(row["mbps_30fps"]), *[_fmt(row["global_psnr"], 2)] * 2,
                     _fmt(row["lpips"], 5), _fmt(row["dists"], 5), _fmt(run["mean_encoder_ms"], 3),
                     _fmt(run["mean_decoder_ms"], 3), _fmt(run["steady_state_fps"], 2)])
    return _table(("Budget", "Wire dtype", "Actual bytes/frame", "Actual bpp", "Mbps@30FPS", "FIXED PSNR",
                   "CLOSED PSNR", "LPIPS", "DISTS", "Encode ms", "Decode ms", "Total FPS"), rows)


def _required(output: Path) -> dict:
    audit_names = ("reproduction", "metrics_parity", "closed_state_audit", "ordered_closed_state_audit",
                   "prefix_audit", "ordered_prefix_audit", "causality_audit", "dynamic_state_audit",
                   "wire_prefix_audit", "wire_roundtrip_audit", "historical_assets_audit", "engineering_audit")
    checks = {name: json_at(output / f"{name}.json")["status"] for name in audit_names}
    if any(status != "PASS" for status in checks.values()):
        raise RuntimeError(f"cannot publish a completed run with a failed required audit: {checks}")
    runs = [json_at(output / f"{name}.json") for name in RUNS]
    checkpoint = json_at(output / "checkpoint_manifest.json")
    if any(run["checkpoint_sha256"] != checkpoint["sha256"] for run in runs):
        raise ValueError("a run did not use the registered C0 snapshot")
    if any(run["model_state_sha256"] != checkpoint["state_dict_sha256"] or
           run.get("state_dict_after_sha256", checkpoint["state_dict_sha256"]) != checkpoint["state_dict_sha256"]
           for run in runs):
        raise ValueError("checkpoint and runtime state_dict evidence disagree")
    if any(run["test_pixel_reads"] or run["training_steps"] for run in runs):
        raise ValueError("Test pixel counter or training update counter is nonzero")
    reads = {key: sum(run["pixel_reads"][key] for run in runs) for key in ("train", "validation", "test")}
    write_json(output / "data_and_weights_audit.json", {
        "status": "PASS", "per_run": dict(zip(RUNS, (run["pixel_reads"] for run in runs), strict=True)),
        "pixel_reads": reads, "test_pixel_reads": 0, "training_steps": 0, "state_dict_sha256": checkpoint["state_dict_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
    })
    return checks


def _bandwidth_table(bands: list[dict], order: str):
    return _table(("计划 Mbps@30FPS", "Wire", "实际 K", "实际 Mbps", "实际 Total bpp", "PSNR", "LPIPS", "DISTS"), [
        [row["target_mbps"], row["wire_dtype"].upper(), row["budget"], _fmt(row["mbps_30fps"], 5),
         _fmt(row["total_bpp"], 8), *_quality(row)] for row in bands if row["order"] == order
    ])


def _capabilities(decisions: dict, runtime: dict, order: str) -> str:
    return _table(("Capability", "Status", "Evidence", "Remaining issue"), [
        ("Experiment Completion", decisions["experiment_completion"], "完整原生 Validation、码流与单卡 H100 矩阵",
         "应用目标带宽和画质门槛待确定"),
        ("Continuous Codec Correctness", decisions["continuous_codec_correctness"],
         "最大预算复现、两精度严格前缀、真实字节、增量和进程隔离", "网络传输留作独立实验"),
        ("Progressive Reconstruction", decisions["progressive_reconstruction"],
         f"O0 与 {order} 的八档质量、相邻增益及逐帧违反率", "低码率绝对画质仍受限"),
        ("Practical Rate–Distortion", decisions["practical_rate_distortion"],
         "C1 等 Payload 和四个候选带宽有真实结果", "应用绝对画质门槛待确定"),
        ("Temporal Causality", decisions["temporal_causality"], "1 帧一块，未来前视和历史状态均为 0",
         "外部传输时序留作独立实验"),
        ("Runtime Performance", decisions["runtime_performance"],
         f'{runtime["minimum_and_deadline_pass"]}/{runtime["wire_configs"]} 个真实码流配置满足 30 FPS 与 P95',
         "按计算精度和码流格式检查部署配置"),
        ("Adaptive Video Streaming Research Readiness", decisions["adaptive_video_streaming_research_readiness"],
         "逐帧可截断、增强包和动态重建已验证", "代表性带宽内画质与真实网络待验证"),
    ])


def _final(output: Path, order: str, progressive: dict, wire, comparison, bands, perf, decisions, runtime) -> None:
    cap_rows = [row for row in comparison if row["comparison"] == "equal_payload_cap"
                and row["method"] == "C0 continuous"]
    core_rows = [row for row in cap_rows if row["order"] == order]
    best = max(cap_rows, key=lambda row: float(row["global_psnr"]))
    high_band = [row for row in bands if row["order"] == order and row["target_mbps"] == "40"]
    observed = (f"{order} 在 C1 Payload 限额下的 PSNR 为 {_geometry_range(core_rows, 'global_psnr')} dB。"
                f"所有连续方案中最高为 {_fmt(best['global_psnr'], 2)} dB（{best['order']}、K{best['budget']}）。"
                f"40 Mbps 下 {order} 为 {_geometry_range(high_band, 'global_psnr')} dB。"
                "C1 Fixed13 等 Payload 参考为 23.25 dB。")
    progression = ("O0、O1、O2 的完整 Validation 最大预算均保持 C0。"
                   "三项质量的汇总邻档判定及逐帧违反率列在下表。"
                   " [全部八档及逐来源结果](ordering_comparison.md)")
    _write(output, "final_report", "# HLA-Tok-C 最终报告",
           "从 V3 `b3b3366` 的 Seed 1 C0 第 10,000 步出发，完成 174 帧原生 1080p Validation。"
           "独立重放的 Global PSNR 为 32.04986 dB，LPIPS-Alex 为 0.09457，DISTS 为 0.05879。"
           "三项指标与 V3 逐值相同。训练步数为 **0**，`test_pixel_reads = 0`。"
           " [复现](reproduction.json) · [权重和数据审计](data_and_weights_audit.json) · [V3 正式 FAIL](../v3/final_report.md)",
           "## 连续表示和时间条件",
           "C0 训练固定使用原生预算 K16320。裁块预算按区域数缩放，同一原生几何未采样预算。"
           "每个 Token 是一个区域内固定 Rank 的 32 维向量，K16320 含 522,240 个标量。"
           "FP32 真实单包为 2,088,988 字节/帧，FP16 和 BF16 为 1,044,508 字节/帧，均包含 28 字节包头。"
           "原生预算编码和公开最大编码后截断各自在 FP32、BF16 下通过 67 档严格前缀。"
           "真实 Payload 前缀、乱序增量、独立进程重建以及所有合法长度的解析通过。"
           " [架构](architecture_audit.md) · [原生前缀](prefix_audit.json) · [码流前缀](wire_prefix_audit.json)"
           " · [独立接收](wire_roundtrip_audit.json)",
           "模型逐帧因果，时间块为单帧。首帧输入就绪即可处理，前视与算法缓冲等待为 0。"
           "FIXED 和 CLOSED 在 C0 上等价，历史同步为 NOT_APPLICABLE。"
           "六条调度在三个真实片段的动态重建逐帧匹配独立解码，恢复高预算的第一帧立即得到对应画质。"
           " [时间因果](causality_audit.json) · [闭环](closed_state_audit.json) · [动态](dynamic_state_audit.json)",
           "## 八档质量与实际传输",
           progression, _progress_text(progressive),
           f"核心表采用静态筛选得到的 {order}。质量和计时均使用 BF16 计算。"
           "FPS 来自真实物理包路径。每组的 FIXED/CLOSED 使用相同无状态结果。",
           _core_table(wire, perf, order),
           "FP16 和 BF16 的格式损失按同顺序、同预算的未序列化 BF16 路径逐一计算。"
           " [三格式精确损失和 24/30/60 FPS 带宽](wire_format_comparison.md)"
           " · [完整三排序矩阵](wire_rate_distortion.csv)",
           "## 等 Payload、候选带宽和速度",
           observed + " 原版 AdapTok 归档走 16 帧输入块，本轮公平比较为 NOT_COMPARABLE。"
           " [等比特口径与结果](equal_rate_analysis.md)",
           _bandwidth_table(bands, order),
           f'真实串行路径的稳定吞吐为 **{runtime["fps_range"]} FPS**，P95 为 '
           f'**{runtime["p95_ms_range"]} ms**，覆盖 FP32/BF16 计算及三种 Wire 格式。'
           " [H100 性能和逐配置门槛](performance_report.md)",
           "## 裁决",
           _capabilities(decisions, runtime, order),
           "证据支持 Scenario B 的静态重排收益和 Scenario D 的浮点表示带宽瓶颈。"
           f"下一步主线是在 C0 + {order} 上做连续表示传输效率实验，保持同一验证队列、逐帧时间口径和"
           "真实总码流预算。当前结果无需启动 Nested Tail Dropout 来获得严格前缀或渐进重建。"
           "是否进一步训练取决于可行码率范围内的画质收益。V3 的 Fixed13 失败记录继续有效。",
           " [七张图及来源](figures/README.md) · [原生 PNG 与裁块](samples/README.md)"
           " · [工程验收](engineering_audit.json) · [复现入口](../../../docs/hla_tok/continuous/README.md)")


def report(output: Path) -> None:
    checks = _required(output)
    decision = json_at(output / "ordering_decision.json")
    order = decision["primary_order"]
    progressive = json_at(output / "progressive_quality.json")
    wire = csv_at(output / "wire_rate_distortion.csv")
    comparison = csv_at(output / "equal_rate_comparison.csv")
    bands = csv_at(output / "bandwidth_constrained_quality.csv")
    perf = csv_at(output / "throughput_benchmark.csv")
    if len(wire) != len(decision["wire_orders"]) * 8 * 3 or len(perf) != 2 * 4 * 8:
        raise ValueError("the measured matrices are incomplete")
    runtime = _perf_report(output, perf)
    decisions = {
        "experiment_completion": "PASS", "continuous_codec_correctness": "PASS",
        "progressive_reconstruction": progressive[order]["status"], "practical_rate_distortion": "NOT_DECIDED",
        "temporal_causality": "PASS", "runtime_performance": runtime["status"],
        "adaptive_video_streaming_research_readiness": "PARTIAL PASS",
    }
    write_json(output / "final_verdict.json", decisions | {
        "primary_order": order, "test_pixel_reads": 0, "training_steps": 0, "v3_historical_status": "FAIL",
        "validation_frames": 174, "runtime": runtime, "audit_status": checks,
        "application_quality_admission": "NOT_DECIDED", "next_experiment": "continuous_representation_rate_efficiency",
    })
    _ordering(output, progressive, decision)
    _wire_comparison(output, wire)
    _equal_analysis(output, comparison)
    _final(output, order, progressive, wire, comparison, bands, perf, decisions, runtime)
    print(f"Experiment completion PASS; practical deployment quality NOT_DECIDED; primary order {order}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
