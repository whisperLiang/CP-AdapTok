"""Read audited artifacts and publish a snapshot without inventing stage results."""

import json
from collections import defaultdict
from datetime import datetime
from os.path import relpath
from zoneinfo import ZoneInfo

from hla_tok import BUDGETS
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file, summarize_quality
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.fixed13 import payload_ledger
from hla_tok.training.checkpoint import read_checkpoint
from scripts.hla_tok.v2.acceptance import record_test_counts
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at
from scripts.hla_tok.v3.experiments import state_of, terminal_step
from scripts.hla_tok.v3.gates import preferred_alternative, quality_pass
from scripts.hla_tok.v3.protocol import CHECKPOINTS, OUTPUT, STAGES

METRICS = ("global_psnr", "lpips", "dists")
PHASE_NAMES = dict(dense="A", continuous="B", quantized="C", progressive="D")
V2 = ROOT / "results/hla_tok/v2"


def _get(path, default=None):
    return json_at(path) if path.is_file() else default


def _training_budget(config, stage, seed):
    name = "progressive_extension.json" if stage == "progressive" else f"{stage}_{seed}_extension.json"
    extension = _get(OUTPUT / "decisions" / name, {})
    additional = extension.get("updates_per_seed", extension.get("updates_per_arm", 0))
    return terminal_step(config, stage) + additional


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
            rows.append({"stage": stage, "seed": seed, "arm": arm, "planned_updates": _training_budget(config, stage, seed),
                         "checkpoint_updates": latest.get("step", 0),
                         "stage_decision": selected.get("status", "PARTIAL PASS"), **counters})
    return rows


def _native_rows():
    rows = []
    for stage in STAGES:
        folder = OUTPUT / stage
        for path in sorted(folder.glob("seed*/*/*/step_*/metrics.json")):
            seed, arm, scope, step = path.parent.parts[-4:]
            for row in json_at(path):
                rows.append({"stage": stage, "seed": seed, "arm": arm, "scope": scope,
                             "step": int(step.removeprefix("step_")), "source": str(path.relative_to(ROOT)), **row})
    return rows


def _historical_rows():
    rows = []
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


def metric_rows():
    return _native_rows() + _historical_rows()


def perf_rows():
    rows = []
    for pattern in ("*/seed*/*/performance/timings.csv", "benchmark/*/timings.csv"):
        for path in sorted(OUTPUT.glob(pattern)):
            rows.extend({"scope": "V3 measured", "source": str(path.relative_to(ROOT)), **row} for row in csv_at(path))
    legacy = V2 / "benchmark/seed1/R2/timings.csv"
    rows.extend(row | {"scope": "V2 timing only; legacy continuous weights and uncalibrated VQ",
                       "source": str(legacy.relative_to(ROOT))} for row in csv_at(legacy))
    return rows


def _closed_evaluation_reads():
    totals = {"train": 0, "validation": 0, "test": 0}
    sources = []
    for path in sorted(OUTPUT.rglob("pixel_reads.json")):
        original = json_at(path)
        parts = {key: original.get(key, original.get(f"{key}_pixel_reads", 0)) for key in totals}
        for key, value in parts.items():
            totals[key] += value
        sources.append({"source": str(path.relative_to(ROOT)), **parts})
    return totals, sources


def _audit_pixels(rows):
    totals, sources = _closed_evaluation_reads()
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


def _phase_states(training):
    active = {(row["stage"], row["seed"]) for row in training}
    pending = {"status": "PARTIAL PASS", "reason": "正式训练进行中，等待终点裁决"}
    blocked = {"status": "BLOCKED", "reason": "等待前置阶段通过"}
    return {f"{prefix}_seed{seed}": _get(state_of(stage, seed), pending if (stage, f"seed{seed}") in active else blocked)
            for seed in (1, 2) for stage, prefix in PHASE_NAMES.items()}


def _overall_status(states, sealed_count):
    statuses = {row["status"] for row in states.values()}
    if statuses == {"PASS"} and sealed_count == 2:
        return "PASS"
    return "FAIL" if "FAIL" in statuses else "PARTIAL PASS"


def _ard_status(states):
    for key in ("B_seed1", "B_seed2"):
        state = states[key]
        if state.get("candidate_quality") == "FAIL" and state["status"] == "BLOCKED":
            return "BLOCKED"
    return "NOT_APPLICABLE"


def _verdicts(training):
    states = _phase_states(training)
    tests = {kind: record_test_counts(OUTPUT / f"{kind}_tests.xml") for kind in ("cpu", "cuda")}
    sealed_count = len(list((OUTPUT / "sealed_test").glob("*/result.json")))
    overall = _overall_status(states, sealed_count)
    return {"status": overall, "stages": states, "v2_d2": "FAIL", "test_status": "PASS" if sealed_count == 2 else "BLOCKED",
            "checkpoint_freeze": _get(OUTPUT / "final_freeze.json"), "tests": tests,
            "strict_reference": _get(OUTPUT / "verify/v2_r2/native_verify.json"),
            "mutations": _get(OUTPUT / "mutations/summary.json"),
            "ard": _ard_status(states), "adaptive_streaming_ready": overall == "PASS"}


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


def _latest_records(rows, fields):
    latest = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key not in latest or row["step"] > latest[key]["step"]:
            latest[key] = row
    return latest.values()


def _latest_observations(metrics, fixed):
    candidates = (row for row in metrics if row["stage"] != "historical" and row.get("budget") in (None, fixed))
    return _latest_records(candidates, ("seed", "arm", "scope", "mode"))


def _render(acceptance, training, observations, audit, output):
    lines = ["# HLA-Tok V3 Reconstruction-First 实验记录", "",
             f"{acceptance['snapshot_time_cst']} 的整体状态为 **{acceptance['status']}**。"
             f"V2 D2 为 **FAIL**。Test 像素读取 **{audit['test_pixel_reads']}**。", "",
             "## 阶段与训练", "",
             "| 阶段 | 状态 | 说明 |", "|---|---|---|"]
    for name, item in acceptance["stages"].items():
        lines.append(f"| {name} | {item['status']} | {item.get('reason') or '见结构化准入记录'} |")
    lines.append("")
    for row in training:
        lines.append(f"- {row['arm']} {row['seed']} 已更新 {row['optimizer_updates']:,}/{row['planned_updates']:,} 步，"
                     f"最近可恢复 {row['checkpoint_updates']:,} 步，累计有效 Train 像素 {row['seen_pixels']:,}。")
    for row in observations:
        scope = "完整 Validation" if row["scope"] == "full" else "趋势验证"
        lines.append(f"- {row['arm']} {row['seed']} 第 {row['step']:,} 步的 {row['frames']} 帧{scope}"
                     f"（{row['mode']}）测得 {_metric_string(row)}。")
    reproduction = relpath(ROOT / "docs/hla_tok/v3/reproduction.md", output)
    lines.extend(("", "## 研究问题", "", _answers(acceptance), "",
                  "三项指标依次为 Global PSNR、LPIPS-Alex 和 DISTS。正式准入使用完整 174 帧原生 Validation。"
                  "率失真表采用真实序列化 payload 字节。", "",
                  "见 [阶段与测试](acceptance.json)、[逐检查点指标](reconstruction_metrics.csv)、"
                  f"[架构对照](architecture_comparison.md) 和 [复现说明]({reproduction})。"))
    return "\n".join(lines) + "\n"


def _dense_selection(config, stages, performance, audit):
    phase = stages["A_seed1"]
    selected = phase.get("selected")
    if not selected:
        return None
    arm = selected["arm"]
    observed = phase["observed"]
    if len(observed) == 2 and all(quality_pass(row, config["dense_thresholds"]) for row in observed.values()):
        if not preferred_alternative(observed["A1"], observed["A0"], config):
            reason = "A0 与 A1 均通过完整验证，A1 的改善低于预注册阈值，按规则选定 A0。"
        elif arm == "A1":
            reason = "A1 达到预注册的配对改善和性能条件。"
        else:
            reason = "A1 达到配对质量改善条件，A0 由预注册性能条件选定。"
    else:
        reason = f"{arm} 达到正式稠密质量门槛。"
    paired_timings = [row for row in performance if "dense/seed1/" in row["source"]]
    modules = audit.get("dense", {}).get(arm, {}).get("modules", {})
    return {"selected_dense_architecture": selected["variant"],
            "selected_checkpoint_path": selected["path"], "selected_checkpoint_sha256": selected["sha256"],
            "selected_encoder_sha256": modules.get("encoder"), "selected_rgb_decoder_sha256": modules.get("rgb_decoder"),
            "selection_reason": reason, "paired_evaluation_summary": observed,
            "paired_dense_performance": paired_timings}


def _bottleneck_result():
    root = OUTPUT / "continuous/seed1/B0/bottleneck"
    markers = [path for scope in ("full", "full_cpu_fp32")
               for path in root.glob(f"step_*/{scope}/measurement.json")]
    if not markers:
        return None
    marker = max(markers, key=lambda path: (path.parent.parent.name, path.parent.name == "full"))
    evidence = json_at(marker)
    result = {row["mode"]: row for row in csv_at(marker.with_name("metrics.csv"))}
    return {"source": str(marker.parent.relative_to(ROOT)), "identity": evidence,
            "paths": {key: result[key] for key in ("phase_a_dense", "b0_dense_through_b0_rgb", "b0_local")}}


def quantization_gap(stages):
    rows = []
    for seed in (1, 2):
        decision = stages[f"C_seed{seed}"]
        immediate = decision.get("immediate_fixed13_minus_bypass")
        if immediate is None:
            continue
        trajectory = decision["continuous_training_trajectory"]
        measured = decision["observed"]
        for key in METRICS:
            direction = 1 if key == "global_psnr" else -1
            bypass = measured[key] - immediate[key]
            rows.append({"seed": seed, "metric": key, "C0_continuous": bypass - trajectory[key],
                         "C1_bypass": bypass, "C1_quantized": measured[key], "actual_bpp": measured["actual_bpp"],
                         "lookup_quality_gain": direction * immediate[key],
                         "training_trajectory_quality_gain": direction * trajectory[key]})
    return rows


def _dense_answer(selection):
    if not selection:
        return "配对裁决等待完整稠密终点。"
    arm = "A0" if selection["selected_dense_architecture"] == "D0" else "A1"
    metrics = selection["paired_evaluation_summary"][arm]
    timing = next((row for row in selection["paired_dense_performance"]
                   if f"/{arm}/" in row["source"] and row["precision"] == "bf16"), None)
    speed = f"，H100 BF16 稠密前向 {float(timing['fps']):.2f} FPS" if timing else ""
    return f"{arm} 在完整原生验证测得 {_metric_string(metrics)}{speed}。"


def _bottleneck_answer(result):
    if not result:
        return "等待同一批 174 帧上的配对路径测量。"
    paths = result["paths"]
    device = result["identity"].get("device", "cuda").upper()
    precision = result["identity"]["precision"].upper()
    return (f"{device} {precision} 在相同 174 帧实测 P0 稠密参照 {_metric_string(paths['phase_a_dense'])}，"
            f"P1 B0 稠密旁路 {_metric_string(paths['b0_dense_through_b0_rgb'])}，"
            f"P2 B0 连续重建 {_metric_string(paths['b0_local'])}。"
            "P1 使用已适应 Local Decoder 输出的 B0 RGB Decoder，差异反映两条路径的实际表现。")


def _progressive_answer(stages):
    measurements = []
    for seed in (1, 2):
        row = stages[f"D_seed{seed}"]
        gains = row.get("gains")
        if gains:
            measurements.append(f"Seed {seed} {row['status']}，K4096→K16320 提升 "
                                f"{gains['base_to_fixed_psnr']:.3f} dB，LPIPS-Alex 降低 "
                                f"{gains['base_to_fixed_lpips']:.4f}，K8192→K16320 提升 "
                                f"{gains['mid_to_fixed_psnr']:.3f} dB。")
    return " ".join(measurements) or "等待双种子固定离散准入和渐进终点。"


def _runtime_answer(performance, required_fps):
    rows = [row for row in performance if "progressive/seed" in row["source"] and row["precision"] == "bf16"]
    if not rows:
        return "等待质量达标的渐进模型和实际性能测量。"
    by_seed = {}
    for row in rows:
        seed = row["source"].split("/progressive/")[1].split("/")[0]
        if float(row["pipeline_fps"]) >= required_fps:
            by_seed.setdefault(seed, []).append(f"K{row['budget']}")
    if not by_seed:
        return "已测预算的 H100 BF16 Offline Pipeline 均低于 30 FPS。"
    return "H100 BF16 Offline Pipeline 达到 30 FPS 的预算为 " + "；".join(
        f"{key} {'、'.join(budgets)}" for key, budgets in by_seed.items()) + "。串行截止时间见性能表。"


def _signed_metric_string(deltas):
    return " / ".join(f"{float(deltas[key]):+.5f}" for key in METRICS)


def _quantization_answers(fixed):
    lookup = fixed.get("immediate_fixed13_minus_bypass")
    if lookup is None:
        return "等待 C0 与 C1 完整验证。", "等待 C1 同权重旁路。"
    trajectory = fixed["continuous_training_trajectory"]
    total = {key: lookup[key] + trajectory[key] for key in METRICS}
    return (f"C1 真实码流相对 C0 的三项原始指标变化为 {_signed_metric_string(total)}。",
            f"同权重查表变化为 {_signed_metric_string(lookup)}。"
            f"C1 旁路相对 C0 的训练轨迹变化为 {_signed_metric_string(trajectory)}。"
            " [三路径量化对照](quantization_gap.csv)。")


def _strict_answer(acceptance):
    strict = acceptance["final_strict_verification"]
    if strict:
        return "、".join(f"{key} {value}" for key, value in strict.items()) + "。"
    fixed = acceptance["fixed_terminal_verification"]
    if fixed:
        return f"C1 终点 FP32 与 BF16 整数前缀验证为 {fixed['status']}。最终渐进权重为 BLOCKED。"
    return "BLOCKED，等待最终训练权重。"


def _answers(acceptance):
    stages = acceptance["stages"]
    selection = acceptance["phase_a_selection"]
    fixed = stages["C_seed1"]
    continuous = stages["B_seed1"]
    metrics = continuous.get("observed")
    quantization, attribution = _quantization_answers(fixed)
    answers = [
        ("A0 与 A1 哪个更适合迁移？",
         selection["selection_reason"] if selection else "配对裁决等待完整验证。"),
        ("V3 的稠密重建质量和效率如何？", _dense_answer(selection)),
        ("R2 能否迁移到 K16320×32 连续序列？",
         f"Seed 1 为 {continuous['status']}，终点 {_metric_string(metrics)}。" if metrics else "等待 B0 终点评价。"),
        ("B0 的剩余差距来自哪里？", _bottleneck_answer(acceptance["phase_b_attribution"])),
        ("ARD 是否必要？",
         "B0 已通过连续门槛，ARD 为 NOT_APPLICABLE。" if continuous["status"] == "PASS" else acceptance["ard"]),
        ("连续与离散的质量差距是多少？", quantization),
        ("即时查表与量化训练轨迹的影响分别是多少？", attribution),
        ("渐进训练的边际收益是多少？", _progressive_answer(stages)),
        ("最终模型的八档严格前缀成立吗？", _strict_answer(acceptance)),
        ("哪些预算达到实时吞吐？", _runtime_answer(acceptance["timing_rows"], acceptance["required_pipeline_fps"])),
        ("是否达到原版 AdapTok 的同码率质量门槛？",
         f"Seed 1 固定离散 {fixed['status']}，Seed 2 固定离散 {stages['C_seed2']['status']}。"),
        ("是否具备 Adaptive Video Streaming 准入资格？",
         "PASS，正式前序和封存评价均已完成。" if acceptance["adaptive_streaming_ready"] else "BLOCKED。"),
    ]
    return "\n".join(f"{index}. **{question}** {result}" for index, (question, result) in enumerate(answers, 1))


def _completed_dense(config, metrics):
    terminal = terminal_step(config, "dense")
    return {f"{row['arm']} {row['seed']}": {
                "quality_status": "PASS" if quality_pass(row, config["dense_thresholds"]) else "FAIL",
                "step": terminal, "metrics": {key: row[key] for key in METRICS},
            } for row in metrics if row["stage"] == "dense" and row["scope"] == "full" and row["step"] == terminal}


def _final_verification():
    paths = sorted(OUTPUT.glob("progressive/seed*/D/verify/native_verify.json"))
    return {path.parts[-4]: json_at(path)["status"] for path in paths}


FRAME_METRICS = ("mse", "psnr", "lpips", "lpips_vgg", "dists", "ms-ssim",
                 "boundary_mse", "interior_mse", "boundary_minus_interior_mse")


def _latest_full_evaluations(metrics):
    candidates = (row for row in metrics if row["stage"] != "historical" and row["scope"] == "full")
    return _latest_records(candidates, ("stage", "arm", "seed"))


def _video_results(frames):
    separate = defaultdict(list)
    for row in frames:
        separate[(row["video"], row["mode"], row["budget"])].append(row)
    results = []
    for (video, mode, budget), items in separate.items():
        observations = [{key: float(item[key]) for key in FRAME_METRICS} for item in items]
        metrics = summarize_quality(observations)
        if mode == "fixed13":
            count = len(items)
            total_bytes = sum(int(item["total_bytes"]) for item in items)
            metrics |= payload_ledger(total_bytes, count * int(budget), count, *NATIVE_SHAPE)
        results.append({"video": video, "mode": mode, "budget": budget} | metrics)
    return results


def _frames_by_budget(frames):
    result = defaultdict(dict)
    for row in frames:
        budget = int(row["budget"])
        key = (row["video"], int(row["frame"]))
        if row["mode"] != "fixed13" or key in result[budget]:
            raise ValueError("progressive frame evidence must contain one Fixed13 row per frame and budget")
        result[budget][key] = row
    return result


def frame_marginal_gains(frames):
    result = []
    by_budget = _frames_by_budget(frames)
    for lower, upper in zip(BUDGETS, BUDGETS[1:]):
        left, right = by_budget[lower], by_budget[upper]
        if not left or left.keys() != right.keys():
            raise ValueError("progressive frame evidence must be matched at every budget")
        for (video, index), earlier in left.items():
            later = right[(video, index)]
            result.append({"video": video, "frame": index, "lower_budget": lower, "upper_budget": upper,
                           "psnr_gain_db": float(later["psnr"]) - float(earlier["psnr"]),
                           "lpips_alex_reduction": float(earlier["lpips"]) - float(later["lpips"]),
                           "dists_reduction": float(earlier["dists"]) - float(later["dists"])})
    return result


def _write_frame_reports(metrics, output):
    videos, gains = [], []
    for measurement in _latest_full_evaluations(metrics):
        source = (ROOT / measurement["source"]).with_name("per_frame.csv")
        if not source.is_file():
            continue
        frames = csv_at(source)
        identity = {key: measurement[key] for key in ("stage", "seed", "arm", "step")}
        videos.extend(identity | row for row in _video_results(frames))
        if measurement["stage"] == "progressive":
            gains.extend(identity | row for row in frame_marginal_gains(frames))
    if videos:
        write_csv(output / "reconstruction_metrics_by_video.csv", videos)
    if gains:
        write_csv(output / "frame_marginal_gains.csv", gains)


def _publish_tables(output, config, acceptance, training, metrics, performance, audit):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "acceptance.json", acceptance)
    write_json(output / "checkpoint_manifest.json", _checkpoint_manifest())
    write_json(output / "data_access_audit.json", audit)
    if training:
        write_csv(output / "training_summary.csv", training)
    write_csv(output / "reconstruction_metrics.csv", metrics)
    write_csv(output / "rate_distortion.csv", [row for row in metrics if row.get("actual_bpp")])
    write_csv(output / "performance.csv", performance)
    quantization = quantization_gap(acceptance["stages"])
    if quantization:
        write_csv(output / "quantization_gap.csv", quantization)
    (output / "architecture_comparison.md").write_text(_comparison(metrics))
    _write_frame_reports(metrics, output)
    observations = _latest_observations(metrics, config["native_budget"])
    (output / "final_report.md").write_text(_render(acceptance, training, observations, audit, output))
    (output / "figures").mkdir(exist_ok=True)


def report(config, v2, output, *, figures=False):
    if v2.resolve() == ROOT.resolve() or not (v2 / "results/hla_tok/v2/acceptance.json").is_file():
        raise ValueError("report requires the separate frozen V2 worktree")
    training, metrics = progress(config), metric_rows()
    audit = _audit_pixels(training)
    performance = perf_rows()
    acceptance = _verdicts(training)
    state_audit = _get(OUTPUT / "live_report/formal_state_audit.json", {})
    acceptance |= {
        "snapshot_time_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "pixel_reads": audit["closed_read_ledgers"],
        "completed_dense_arms": _completed_dense(config, metrics),
        "resume_check": _get(OUTPUT / "resume_audit.json"),
        "old_assets": _get(OUTPUT / "legacy_invariance.json"),
        "phase_a_selection": _dense_selection(config, acceptance["stages"], performance, state_audit),
        "phase_b_attribution": _bottleneck_result(),
        "final_strict_verification": _final_verification(),
        "fixed_terminal_verification": _get(OUTPUT / "engineering_audit.json", {}).get("fixed_terminal_verification"),
        "required_pipeline_fps": config["performance"]["pipeline_fps"],
        "timing_rows": [row for row in performance if row["source"].startswith("results/hla_tok/v3/progressive/")],
    }
    _publish_tables(output, config, acceptance, training, metrics, performance, audit)
    if figures:
        from scripts.hla_tok.v3.figures import figures_from

        figures_from(metrics, config, output / "figures")
    return acceptance
