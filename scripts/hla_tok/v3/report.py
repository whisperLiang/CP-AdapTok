"""Read audited artifacts and publish a snapshot without inventing stage results."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.training.checkpoint import read_checkpoint
from scripts.hla_tok.v2.acceptance import record_test_counts
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at
from scripts.hla_tok.v3.experiments import state_of, terminal_step
from scripts.hla_tok.v3.gates import quality_pass
from scripts.hla_tok.v3.protocol import CHECKPOINTS, OUTPUT, STAGES

METRICS = ("global_psnr", "lpips", "dists")
V2 = ROOT / "results/hla_tok/v2"


def _get(path, default=None):
    return json_at(path) if path.is_file() else default


def progress(config):
    rows = []
    for stage in STAGES:
        for path in sorted((OUTPUT / stage).glob("seed*/*/training.json")):
            seed, arm = path.parent.parts[-2:]
            info = json_at(path)
            latest = _get(path.parent / "latest.json", {})
            selected = _get(state_of(stage, int(seed.removeprefix("seed"))), {})
            counters = {(f"last_invocation_{key}" if key.startswith("peak_") else key): value
                        for key, value in info.items() if not isinstance(value, dict)}
            rows.append({"stage": stage, "seed": seed, "arm": arm, "planned_updates": terminal_step(config, stage),
                         "checkpoint_updates": latest.get("step", 0),
                         "stage_decision": selected.get("status", "PARTIAL PASS"), **counters})
    return rows


def metric_rows():
    rows = []
    for stage in STAGES:
        folder = OUTPUT / stage
        for path in sorted(folder.glob("seed*/*/*/step_*/metrics.json")):
            seed, arm, scope, step = path.parent.parts[-4:]
            for row in json_at(path):
                rows.append({"stage": stage, "seed": seed, "arm": arm, "scope": scope,
                             "step": int(step.removeprefix("step_")), "source": str(path.relative_to(ROOT)), **row})
    replay = _get(OUTPUT / "baseline" / "summary.json")
    if replay:
        rows.append({"stage": "historical", "seed": "seed1", "arm": "V2 R2 replay", "scope": "full",
                     "step": 2000, "mode": "continuous", **replay})
    acceptance = json_at(V2 / "acceptance.json")
    for row in acceptance["D2"]["paired_comparisons"]:
        if row["arm"] == "R0":
            rows.append(row | {"stage": "historical", "seed": "seed1", "arm": "V2 R0", "scope": "full",
                               "step": 2000, "mode": "continuous"})
    rows.extend(row | {"stage": "historical", "arm": f"{row['arm']} frozen encoder", "scope": "trend", "mode": "dense"}
                for row in csv_at(V2 / "diag0/oracles/metrics.csv"))
    rows.extend(row | {"stage": "historical", "arm": "HLA-Tok R0 Fixed13", "scope": "full", "mode": "fixed13"}
                for row in csv_at(ROOT / "results/hla_tok/phase1/fixed/validation_metrics.csv"))
    rows.extend(row | {"stage": "historical", "scope": "full", "arm": row["method"]}
                for row in csv_at(ROOT / "results/hla_tok/baselines/metrics.csv")
                if row["frames"] == "174" and row["global_psnr"])
    return rows


def perf_rows():
    rows = []
    for pattern in ("*/seed*/*/performance/timings.csv", "benchmark/*/timings.csv"):
        for path in sorted(OUTPUT.glob(pattern)):
            rows.extend({"scope": "V3 measured", "source": str(path.relative_to(ROOT)), **row} for row in csv_at(path))
    legacy = V2 / "benchmark/seed1/R2/timings.csv"
    rows.extend(row | {"scope": "V2 timing only; legacy continuous weights and uncalibrated VQ",
                       "source": str(legacy.relative_to(ROOT))} for row in csv_at(legacy))
    return rows


def _audit_pixels(rows):
    totals = {"train": 0, "validation": 0, "test": 0}
    sources = []
    for path in sorted(OUTPUT.rglob("pixel_reads.json")):
        original = json_at(path)
        parts = {key: original.get(key, original.get(f"{key}_pixel_reads", 0)) for key in totals}
        for key, value in parts.items():
            totals[key] += value
        sources.append({"source": str(path.relative_to(ROOT)), **parts})
    for path in sorted(OUTPUT.rglob("train_reads_after_*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]
        totals["train"] += sum(row["train"] for row in records)
        totals["test"] += sum(row["test"] for row in records)
    preflight = _get(OUTPUT / "preflight.json", {})
    return {"closed_read_ledgers": {f"{key}_pixel_reads": count for key, count in totals.items()},
            "train_completed_optimizer_updates": sum(row["optimizer_updates"] for row in rows),
            "test_pixel_reads": totals["test"], "test_sealed": not (OUTPUT / "final_freeze.json").exists(),
            "note": "A live optimization segment flushes its read ledger on stop; prefetched Train reads are separate.",
            "manifests": preflight.get("manifest_sha256"), "cohort": preflight.get("cohort"), "sources": sources}


def _checkpoint_manifest():
    rows = []
    for path in sorted(CHECKPOINTS.rglob("step_*.pt")):
        info = read_checkpoint(path).state
        rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                     "step": info["step"], "data_cursor": info["data_cursor"],
                     "architecture": info["counters"].get("architecture"), "identity": info["identity"],
                     "sample_chain": info["counters"].get("sample_chain"),
                     "is_setup_audit": path.is_relative_to(CHECKPOINTS / "audit")})
    return rows


def _verdicts(training):
    states = {}
    stage_names = {"dense": "A", "continuous": "B", "quantized": "C", "progressive": "D"}
    for seed in (1, 2):
        for stage, prefix in stage_names.items():
            active = any(row["stage"] == stage and row["seed"] == f"seed{seed}" for row in training)
            default = {"status": "PARTIAL PASS" if active else "BLOCKED",
                       "reason": "正式训练未完成，不给出质量准入结论" if active else "前置阶段未取得 PASS"}
            states[f"{prefix}_seed{seed}"] = _get(state_of(stage, seed), default)
    tests = {kind: record_test_counts(OUTPUT / f"{kind}_tests.xml") for kind in ("cpu", "cuda")}
    native = _get(OUTPUT / "verify/v2_r2/native_verify.json")
    mutation = _get(OUTPUT / "mutations/summary.json")
    final = _get(OUTPUT / "final_freeze.json")
    sealed = list((OUTPUT / "sealed_test").glob("*/result.json"))
    passed = all(row["status"] == "PASS" for row in states.values())
    stopped = any(row["status"] == "FAIL" for row in states.values())
    overall = "PASS" if passed and len(sealed) == 2 else ("FAIL" if stopped else "PARTIAL PASS")
    return {"status": overall, "stages": states, "v2_d2": "FAIL", "test_status": "PASS" if len(sealed) == 2 else "BLOCKED",
            "checkpoint_freeze": final, "tests": tests, "strict_reference": native, "mutations": mutation,
            "ard": "NOT_APPLICABLE", "adaptive_streaming_ready": passed and len(sealed) == 2}


def _metric_string(row):
    return " / ".join(f"{float(row[key]):.5f}" for key in METRICS)


def _historical_reference(row):
    arm = row["arm"]
    if arm in {"V2 R0", "V2 R2 replay", "O0 frozen encoder", "O1 frozen encoder", "HLA-Tok R0 Fixed13"}:
        return True
    if arm == "AIP-VTok Joint V1":
        return row["budget"] == "16320" and row["history"] == "CLOSED"
    return (arm == "Original AdapTok" and row["precision"] == "fp32" and row["stride"] == "128"
            and round(float(row["actual_bpp"]), 6) in (0.089655, 0.112069))


def _comparison(rows):
    candidates = [row for row in rows if (row["scope"] == "full" if row["stage"] != "historical"
                                          else _historical_reference(row))]
    table = ["| 模型 / 观察 | 更新 | 帧数 | PSNR / LPIPS-Alex / DISTS | Actual bpp |",
             "|---|---:|---:|---|---:|"]
    for row in candidates:
        rate = f"{float(row['actual_bpp']):.6f}" if row.get("actual_bpp") else "—"
        label = f"{row['arm']} {row.get('mode', '')}" + (f" K{row['budget']}" if row.get("budget") else "")
        table.append(f"| {label} | {row.get('step', row.get('updates', '—'))} | {row['frames']} | "
                     f"{_metric_string(row)} | {rate} |")
    return "\n".join(["# HLA-Tok V3 架构对照", "",
                      "D0 保持旧键；D1 仅在 1/8 和 1/4 尺度各附加一个恒等初始化 LiteBlock。"
                      "稠密路径仅保留 Encoder coarse → RGB Decoder，不经过发送端 Tokenizer，亦没有码率。"
                      "B0 采用原 R2 / Local Decoder；只有实际 Fixed13 字节可进入 V3 率失真表。", "",
                      *table, "",
                      "O0/O1 是旧的 9 帧、冻结 Encoder 稠密适配诊断；不能当作端到端训练或 174 帧评价。"]) + "\n"


def _render(acceptance, training, metrics, audit):
    lines = ["# HLA-Tok V3 Reconstruction-First 实验记录", "",
             f"快照时间：{acceptance['snapshot_time_cst']}。整体：**{acceptance['status']}**。"
             f" V2 D2 保持 **FAIL**；Test 像素读取 **{audit['test_pixel_reads']}**。", "",
             "## 阶段与训练", "",
             "| 阶段 | 状态 | 说明 |", "|---|---|---|"]
    for name, item in acceptance["stages"].items():
        lines.append(f"| {name} | {item['status']} | {item.get('reason') or '见结构化准入记录'} |")
    for row in training:
        lines.append(f"- {row['arm']} {row['seed']}：实际 {row['optimizer_updates']:,}/{row['planned_updates']:,} 更新；"
                     f"最近可恢复 {row['checkpoint_updates']:,}；有效 Train 像素 {row['seen_pixels']:,}。")
    for arm, result in acceptance["completed_dense_arms"].items():
        lines.append(f"- {arm} 终点质量 **{result['quality_status']}**。单臂达标不等于 Phase A 配对准入。")
    latest = {}
    for row in metrics:
        if row["stage"] == "dense":
            latest[(row["arm"], row["scope"])] = row
    for (_, scope), row in latest.items():
        lines.append(f"- {row['arm']} 第 {row['step']:,} 步，{row['frames']} 帧 "
                     f"{'完整 Validation' if scope == 'full' else '趋势诊断'}：{_metric_string(row)}。")
    benchmark = OUTPUT / "benchmark" / "A0_step12000" / "timings.csv"
    if benchmark.is_file():
        timing, = (row for row in csv_at(benchmark) if row["precision"] == "bf16")
        lines.append(f"- A0 终点 H100 BF16 稠密前向：{float(timing['fps']):.2f} FPS，"
                     f"P95 {float(timing['p95_ms']):.3f} ms，峰值分配 {float(timing['peak_allocated_mib']):.0f} MiB。"
                     " 此处没有 Token、码率或网络流水线。")
    replay = _get(OUTPUT / "baseline" / "replay.json", {})
    if replay:
        lines.extend(["", "## 历史复现与准入边界", "",
                      f"旧 R2 2000 步检查点在同一 174 帧上逐指标精确复现：{_metric_string(replay['replayed'])}。"
                      " 固定阈值仍为 24.92307 dB / 0.20373 / 0.09173；旧 R2 没有已验收的离散码流。"])
    lines.extend(["", "## 十项研究问题", "",
                  _answers(latest, acceptance), "",
                  "PSNR 为 Global PSNR；LPIPS 为 LPIPS-Alex。9 帧观察不参与任何晋级。质量不确定性按视频重采样，"
                  "不把相邻帧当独立样本。码率表排除稠密模型和浮点连续表示；历史 FPS 不用于 V3 实时性裁决。", "",
                  "见 [阶段与测试 JSON](acceptance.json)、[逐检查点指标](reconstruction_metrics.csv)、"
                  "[架构对照](architecture_comparison.md) 和 [复现说明](../../../docs/hla_tok/v3/reproduction.md)。"])
    return "\n".join(lines) + "\n"


def _answers(latest, acceptance):
    dense_full = [row for (_, scope), row in latest.items() if scope == "full"]
    measured = "；".join(f"{row['arm']} {row['step']} 步 {_metric_string(row)}" for row in dense_full)
    stages = acceptance["stages"]
    fixed = stages["C_seed1"]
    progressive = stages["D_seed1"]
    perf = progressive.get("performance") or fixed.get("performance")
    dense_pass = any(record["quality_status"] == "PASS" for record in acceptance["completed_dense_arms"].values())
    answers = [
        f"原生稠密可恢复性：{measured or '没有完成 V3 全帧评价'}；以 Phase A 终点裁决为准。",
        "V2 感知质量不足是否解决：" + ("端到端稠密臂已有达标终点；这不能证明连续或离散 Token 也已达标。" if
                                     dense_pass else "尚无达到质量门槛的稠密终点。"),
        "连续 Token 损失：" + (_metric_string(stages["B_seed1"]["observed"]) if "observed" in stages["B_seed1"]
                               else "尚无匹配的 B0 全帧读数。"),
        "是否需要 ARD：未启用；只有 B0 的真实失配与成对 Local Decoder 诊断能够触发。",
        "VQ 即时与轨迹误差：" + (str(fixed["immediate_fixed13_minus_bypass"]) + " / "
                                  + str(fixed["continuous_training_trajectory"]) if
                                  "immediate_fixed13_minus_bypass" in fixed else "未测；连续 R2 不是量化控制。"),
        "原 Phase 1 bpp 质量门槛：Seed 1 固定离散阶段 " + fixed["status"] + "。",
        "渐进式新增 Token 的边际收益：Seed 1 渐进阶段 " + progressive["status"] + "。",
        "H100 K16320 性能：" + (f"{perf['pipeline_fps']:.2f} 离线 FPS，严格串行 P95 "
                                 f"{perf['serial_p95_ms']:.2f} ms。" if perf else "未测通过质量的 V3 离散模型。"),
        "跨种子一致性：Seed 1 / Seed 2 固定离散 " + fixed["status"] + " / " + stages["C_seed2"]["status"] + "。",
        "自适应传输准入：" + ("满足预注册的本地编解码条件；不等同于真实网络端到端吞吐。" if
                                acceptance["adaptive_streaming_ready"] else "未获得。"),
    ]
    return "\n".join(f"{index}. {text}" for index, text in enumerate(answers, 1))


def report(config, v2, output, *, figures=False):
    if v2.resolve() == ROOT.resolve() or not (v2 / "results/hla_tok/v2/acceptance.json").is_file():
        raise ValueError("report requires the separate frozen V2 worktree")
    training, metrics = progress(config), metric_rows()
    audit = _audit_pixels(training)
    terminal = terminal_step(config, "dense")
    complete = {f"{row['arm']} {row['seed']}": {
                    "quality_status": "PASS" if quality_pass(row, config["dense_thresholds"]) else "FAIL",
                    "step": terminal, "metrics": {key: row[key] for key in METRICS},
                } for row in metrics if row["stage"] == "dense" and row["scope"] == "full" and row["step"] == terminal}
    acceptance = _verdicts(training) | {
        "snapshot_time_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "pixel_reads": audit["closed_read_ledgers"],
        "completed_dense_arms": complete,
        "resume_check": _get(OUTPUT / "resume_audit.json"),
        "old_assets": _get(OUTPUT / "legacy_invariance.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "acceptance.json", acceptance)
    write_json(output / "checkpoint_manifest.json", _checkpoint_manifest())
    write_json(output / "data_access_audit.json", audit)
    if training:
        write_csv(output / "training_summary.csv", training)
    write_csv(output / "reconstruction_metrics.csv", metrics)
    write_csv(output / "rate_distortion.csv", [row for row in metrics if row.get("actual_bpp")])
    write_csv(output / "performance.csv", perf_rows())
    (output / "architecture_comparison.md").write_text(_comparison(metrics))
    (output / "final_report.md").write_text(_render(acceptance, training, metrics, audit))
    (output / "figures").mkdir(exist_ok=True)
    if figures:
        from scripts.hla_tok.v3.figures import figures_from

        figures_from(metrics, config, output / "figures")
    return acceptance
