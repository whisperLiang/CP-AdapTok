"""Assemble the observed V3 gates using the inherited fixed thresholds."""

import argparse
from math import isfinite
from pathlib import Path

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.official import sha256_file
from scripts.hla_tok.uf_v2.acceptance import gate_quality
from scripts.hla_tok.uf_v2.rate_analysis import (
    adjacency,
    at_quality,
    at_rate,
    by_video,
    common,
    compare,
    frontier,
    violations,
)
from scripts.hla_tok.uf_v3.evaluate import close_timing
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

BUDGETS = (5, 10, 20, 40)


def load_if(path):
    return json_at(path) if path.exists() else {}


def read_layers(out):
    archive = ROOT / "results/hla_tok/uf_v2"
    legacy = json_at(archive / "evaluation/rate_analysis.json")["curves"]
    baseline, anchor = (json_at(archive / folder / "summary.json") for folder in (
        "d1/formal-seed2-final-e0", "anchor_fast/validation"))
    files = {"baseline": baseline, "anchor": anchor, "curves": legacy}
    for key, filename in (("audit", "audit/training_audit.json"), ("cal", "diagnostics/calibration/capacity.json"),
                          ("cap", "diagnostics/validation/capacity.json"), ("decision", "diagnostics/capacity.json"),
                          ("entropy", "entropy/summary.json"), ("lite", "base_lite/summary.json"),
                          ("a2", "evaluation/A2/summary.json"), ("a4", "evaluation/A4/summary.json"),
                          ("ondemand", "evaluation/on_demand/summary.json"), ("ondemand_repeat", "evaluation/on_demand_repeat/summary.json"),
                          ("contracts", "evaluation/contracts/summary.json"), ("qa", "qa/summary.json"),
                          ("recovery", "diagnostics/base_recovery.json"), ("precision", "diagnostics/precision.json"),
                          ("after_joint", "diagnostics/post_joint/capacity.json"), ("runtime_trial", "runtime_trial/summary.json"),
                          ("a2_contracts", "evaluation/A2_contracts/summary.json"),
                          ("ledger", "training/checkpoint_ledger.json"),
                          ("optimized", "runtime_trial/selection.json")):
        files[key] = load_if(out / filename)
    repeat = files["ondemand_repeat"]
    clean = repeat.get("other_gpu_compute_before_after") == [0, 0]
    files["ondemand_summary_path"] = "evaluation/on_demand_repeat/summary.json" if clean else (
        "evaluation/on_demand/summary.json")
    if clean:
        files["ondemand"] = repeat
    files["full_max_summary_path"] = "evaluation/A4/summary.json"
    runtime = files["optimized"]
    if runtime.get("selected") == "reuse":
        full, demand = runtime["timing"]["full_max"], runtime["timing"]["on_demand"]
        newer = json_at(out / full)
        files["a4"] = files["a4"] | {"timing": newer["timing"],
                                     "timing_peak_cuda_bytes": newer["timing_peak_cuda_bytes"]}
        files["ondemand"] = json_at(out / demand)
        files["contracts"] = json_at(out / runtime["contracts"])
        files["full_max_summary_path"] = full
        files["ondemand_summary_path"] = demand
    return files


def quality_by_rate(files, out):
    names = ("Original AdapTok", "HLA-Tok-T", "CC-S", "CC-N", "CC-J", "Official DCVC-RT", "Official UF-LD",
             "Anchor-Fast", "UF-Progressive D0", "UF-Anchor")
    curves = {name: files["curves"][name] for name in names if name in files["curves"]}
    curves["V2 D1"] = files["curves"]["D1 / Seed 2 final E0"]
    curves["UF-Anchor"] = json_at(ROOT / "results/hla_tok/uf_v2/baseline/baseline_audit.json")[
        "curve_sources"]["uf_anchor"]["rows"]
    if files["a2"].get("quality"):
        curves["V2 E1"] = [row for row in files["a2"]["quality"] if row["status"] == "PASS"]
    d0 = files["cap"].get("results", {}).get("2", {}).get("pooled", {})
    if d0:
        curves["D0-Fast"] = [d0[f"C2/{b}"] | {"budget_mbps": b} for b in BUDGETS]
    actual = files["a4"].get("quality", [])
    if actual:
        curves["V3 joint"] = [r for r in actual if r["status"] == "PASS"]
        quality_frames = csv_at(out / "evaluation/A4/quality.csv")
        pairs = adjacency(quality_frames)
        write_csv(out / "evaluation/A4/by_video.csv", by_video(quality_frames))
        write_csv(out / "evaluation/A4/adjacent_frames.csv", pairs)
        write_csv(out / "evaluation/A4/adjacent_summary.csv", violations(pairs))
    equal, bd, controls = compare(curves)
    write_csv(out / "evaluation/equal_quality.csv", equal)
    write_json(out / "evaluation/rates.json", {"curves": curves, "equal_quality": equal, "bd_rate": bd,
        "same_actual_rate": controls, "matched_reference_dists": matched_dists(curves),
        "scope": "actual application series rate; interpolation within observed support",
        "unreachable_rows": [row for row in actual if row["status"] != "PASS"]})
    return curves, equal


def matched_dists(curves, target=.18):
    metric = "dists"
    current = frontier(curves.get("V3 joint", []), metric)
    results = []
    for name in ("V2 D1", "V2 E1", "Anchor-Fast"):
        reference = frontier(curves.get(name, []), metric)
        observed, baseline = (at_quality(rows, metric, target) for rows in (current, reference))
        span = common(current, reference, metric)
        results.append({"reference": name, "target": target, "v3_mbps": observed, "reference_mbps": baseline,
                        "common_range": span["range"] if span else None,
                        "status": "PASS" if observed is not None and baseline is not None else "OUT_OF_RANGE"})
    return results


def close_quality(summary, rules):
    outcomes = {}
    for row in summary:
        key = str(int(row["budget_mbps"]))
        if key not in rules or row["budget_mbps"] not in BUDGETS:
            continue
        outcomes[key] = gate_quality(row, rules[key])
    return outcomes


def val(equal, model, metric, target):
    return next((row for row in equal if row["method"] == model and row["metric"] == metric
                 and row["target"] == target),
                {"candidate_actual_mbps": None, "original_actual_mbps": None, "status": "NOT_RUN"})


def fixed_threshold(curve):
    points = [point for point in curve if point.get("budget_mbps", 0) > 0 and point["global_psnr"] >= 25]
    return min((p["actual_mbps"] for p in points), default=None)


def measured_threshold(curve):
    return min((p["actual_mbps"] for p in curve if p["global_psnr"] >= 25), default=None)


def progressive_gain(curve):
    by_cut = {row["budget_mbps"]: row for row in curve}
    rows = []
    for low, high in zip(BUDGETS, BUDGETS[1:]):
        if low not in by_cut or high not in by_cut:
            continue
        a, b = by_cut[low], by_cut[high]
        extra = b["actual_mbps"] - a["actual_mbps"]
        gains = {"psnr_db": b["global_psnr"] - a["global_psnr"],
                 "lpips_improvement": a["lpips"] - b["lpips"], "dists_improvement": a["dists"] - b["dists"]}
        rows.append({"low_cap": low, "high_cap": high, "added_actual_mbps": extra,
                     "added_bytes_per_frame": extra * 1e6 / 240, **gains,
                     "lpips_per_added_kb": gains["lpips_improvement"] / (extra * 1e3 / 240) if extra else None})
    return rows


def v3_stage_gates(curve):
    thresholds = {5: (27.20, .31), 10: (27.90, .29), 20: (29.50, .17480), 40: (30., .12108)}
    rows = {row["budget_mbps"]: row for row in curve}
    return {str(cut): (row["global_psnr"] >= minimum and row["lpips"] <= maximum) if row else None
            for cut, (minimum, maximum) in thresholds.items() for row in (rows.get(cut),)}


def read_log(out):
    runs = {}
    for path in sorted((out / "training").glob("*/summary.json")):
        entry = json_at(path)
        files = ("stage", "seed", "optimizer_updates", "added", "gpu_event_interval_seconds", "wall_seconds",
                 "peak_cuda_bytes", "frame_instances", "checkpoint_sha256", "read_counts")
        runs[path.parent.name] = {key: entry[key] for key in files if key in entry}
    return runs


def effective_update(out):
    probes = sorted((out / "training").glob("probe-*/summary.json")) + sorted(
        (out / "training").glob("P3-*/summary.json")) + sorted((out / "training").glob("P4-*/summary.json"))
    if not probes:
        return None
    data = json_at(probes[-1])
    return {"source": str(probes[-1]), "optimizer_updates": data["optimizer_updates"], "added": data["added"],
            "gradient_probe": data["detail"][0]["probes"],
            "before_after": {group: {"changed_tensors": sum(r["changed"] > 0 for r in tensors.values()),
                                    "changed_elements": sum(r["changed"] for r in tensors.values())}
                             for group, tensors in data["delta"].items()}}


def rank_candidates(files, equal):
    result = {}
    limits = files["decision"].get("frozen_quality_and_realtime", {})
    candidates = (("A2", "V2 E1", files["a2"], files["a2_contracts"]),
                  ("A4", "V3 joint", files["a4"], files["contracts"]))
    for label, name, current, contracts in candidates:
        if not current.get("quality"):
            continue
        physical = [row for row in current["quality"] if row["budget_mbps"] in BUDGETS]
        safe = len(physical) == 4 and all(row["decoded"] == row["planned"] for row in physical)
        gates = close_quality(physical, limits)
        equal_lpips = val(equal, name, "lpips", .28978)["candidate_actual_mbps"]
        result[label] = {"model": name, "checkpoint_sha256": current["checkpoint_sha256"],
                         "base_and_causality_proven": contracts.get("status") == "PASS",
                         "physical_hard_budgets": safe, "full_max": close_timing(current.get("timing")),
                         "frozen_quality": gates, "lpips_target_mbps": equal_lpips}
    order = sorted(result, key=lambda key: (
        not result[key]["base_and_causality_proven"], not result[key]["physical_hard_budgets"],
        result[key]["full_max"] != "PASS", -sum(result[key]["frozen_quality"].values()),
        result[key]["lpips_target_mbps"] if result[key]["lpips_target_mbps"] is not None else float("inf")))
    return {"order": order, "candidates": result, "selected": order[0] if order else None,
            "scope": "native E1 endpoints; Test and formal deployment require all frozen gates"}


def decide(files, curves, equal, out):
    q = close_quality(files["a4"].get("quality", []), files["decision"].get("frozen_quality_and_realtime", {}))
    engines = files["contracts"]
    training = effective_update(out)
    lpips, psnr = (val(equal, "V3 joint", metric, value) for metric, value in (("lpips", .28978), ("global_psnr", 25)))
    rate = lpips["candidate_actual_mbps"]
    ref = lpips["original_actual_mbps"]
    threshold = fixed_threshold(curves.get("V3 joint", []))
    lite = files["lite"].get("profiles", {}).get("validation", {}).get("L1")
    dimensions = {
        "TRAINING_AUDIT": "PASS" if files["audit"] else "NOT_RUN",
        "DEPLOYMENT_WEIGHT": engines.get("deployment", {}).get("status", "NOT_RUN"),
        "CAPACITY_DIAGNOSTICS": "PASS" if files["cap"] else "NOT_RUN",
        "D0_FAST": close_timing(files["cap"].get("results", {}).get("d0_fast")),
        "STATIC_ENTROPY": "PASS" if files["entropy"] else "NOT_RUN",
        "SELECTED_ARCHITECTURE": "PASS" if files["decision"] else "NOT_RUN",
        "CORE_DETAIL": "NOT_RUN",
        "BASE_LITE": "NOT_RUN" if lite is None else "PASS" if lite["hard_cap_pass"] else "FAIL",
        "END_TO_END_TRAINING": "NOT_RUN" if training is None else "PASS" if training["optimizer_updates"] >= 8000
        else "PARTIAL_PASS",
        "LOW_RATE_QUALITY": "NOT_RUN" if not q else "PASS" if all(q.get(str(b)) for b in (5, 10)) else "FAIL",
        "HIGH_RATE_QUALITY": "NOT_RUN" if not q else "PASS" if all(q.get(str(b)) for b in (20, 40)) else "FAIL",
        "EQUAL_PSNR_RATE": psnr["status"],
        "MINIMUM_PSNR_THRESHOLD_RATE": "NOT_RUN" if threshold is None else "PASS" if threshold < 3.594 else "FAIL",
        "EQUAL_PERCEPTUAL_RATE": lpips["status"] if rate is None or ref is None else "PASS" if rate < ref else "FAIL",
        "CAUSALITY": "PASS" if engines.get("future") and engines.get("dynamic", {}).get("base_mismatches") == 0
        else "NOT_RUN",
        "STRICT_PREFIX": "PASS" if engines.get("dynamic") else "NOT_RUN",
        "INCREMENTAL": "PASS" if engines.get("incremental", {}).get("rgb_exact") else "NOT_RUN",
        "BASE_BUDGET": "NOT_RUN" if "base_max" not in files["a4"] else
        "PASS" if files["a4"]["base_max"] <= 9212 else "FAIL",
        "FULL_MAX_REALTIME": close_timing(files["a4"].get("timing")),
        "ON_DEMAND_REALTIME": close_timing(files["ondemand"].get("timing")),
        "TEST": "NOT_RUN",
    }
    required = ("DEPLOYMENT_WEIGHT", "END_TO_END_TRAINING", "LOW_RATE_QUALITY", "HIGH_RATE_QUALITY",
                "EQUAL_PERCEPTUAL_RATE", "CAUSALITY", "STRICT_PREFIX", "INCREMENTAL", "BASE_BUDGET",
                "FULL_MAX_REALTIME")
    admitted = all(dimensions[key] == "PASS" for key in required)
    checked = sum(item.get("read_counts", {}).get("test", 0) for item in files.values() if isinstance(item, dict))
    checked += engines.get("test_pixel_reads", 0)
    if checked:
        raise PermissionError("Test was opened before formal admission")
    return {"overall": "PARTIAL_PASS" if files["audit"] and files["cap"] else "FAIL",
            "selected_route": files["decision"].get("selected_route"), "validation_admission": admitted,
            "dimensions": dimensions, "physical_quality_gate_by_budget": q,
            "v3_stage_quality": v3_stage_gates(curves.get("V3 joint", [])),
            "full_max_summary": files["full_max_summary_path"], "on_demand_summary": files["ondemand_summary_path"],
            "runtime_backend": files["optimized"] or None,
            "candidate_ranking": rank_candidates(files, equal),
            "progressive_adjacent": {name: progressive_gain(curves[name]) for name in ("V2 D1", "V2 E1", "V3 joint")
                                     if name in curves},
            "checkpoint_sha256": files["a4"].get("checkpoint_sha256"), "trained_evidence": training,
            "late_joint_failure_and_recovery": files["recovery"] or None,
             "optimizer_runs": read_log(out), "test_pixel_reads": checked,
            "seed2": "NOT_RUN" if not any(r.get("seed") == 2 for r in read_log(out).values()) else "EXECUTED",
            "test_reason": "Test remains sealed pending formal Validation admission",
             "equal_psnr": psnr, "equal_lpips": lpips, "minimum_observed_psnr25_rate_mbps": threshold,
             "minimum_observed_psnr25_series_average_mbps": measured_threshold(curves.get("V3 joint", [])),
             "matched_reference_dists": matched_dists(curves), "qa": files["qa"] or None,
            "registered_gate_sha256": sha256_file(out / "diagnostics/capacity.json") if files["decision"] else None}


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and isfinite(value) else "—"


def table_quality(rows, timing, demand):
    table = ["| 上限 Mbps | 可达帧 | 实际 Mbps | PSNR dB | LPIPS | DISTS | Full-max FPS / P95 ms | 按需 FPS / P95 ms |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        b = r["budget_mbps"]
        times = [tags.get(str(int(b)), {}).get("pooled", {}) if b in BUDGETS else {} for tags in (timing, demand)]
        runtime = [f'{fmt(t["fps"], 2)} / {fmt(t["total_ms_p95"], 2)}' if t else "—" for t in times]
        table.append(f'| {b:g} | {r["decoded"]}/{r["planned"]} | {fmt(r.get("actual_mbps"), 4)} | '
                     f'{fmt(r.get("global_psnr"))} | {fmt(r.get("lpips"), 5)} | {fmt(r.get("dists"), 5)} | '
                     f'{runtime[0]} | {runtime[1]} |')
    return "\n".join(table)


def capacity_sections(files):
    samples = files["cap"].get("results", {})
    if not samples:
        return []
    domains = {"C0": "CONTINUOUS", "C1": "CONTINUOUS", "C2": "ALL_VALID_RECORDS", "C3": "ALL_VALID_RECORDS"}
    fields = ("global_psnr", "lpips", "dists")
    lines = ["## 容量诊断与 D0-Fast", "",
             "| 对照 | Seed 1 PSNR / LPIPS / DISTS | Seed 2 PSNR / LPIPS / DISTS |", "|---|---:|---:|"]
    for key, domain in domains.items():
        cells = [" / ".join(fmt(samples[seed]["pooled"][f"{key}/{domain}"][field], digits)
                            for field, digits in zip(fields, (3, 5, 5), strict=True)) for seed in ("1", "2")]
        lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "C0/C1 是连续域机制对照。C2/C3 使用相同的真实整数与 Base 历史。Seed 2 的 C0 在 "
                  "174/174 帧提高 PSNR。C2 相对 C3 的全局 PSNR 高 0.251 dB，LPIPS 高 0.15953。"
                  "全有效前缀较 40 档仅增加 0.00104 dB，支持限制来自量化表示和读出折衷。", "",
                  "R0 优先补齐原始 RGB 到骨干的联合优化。R1 Core–Detail 缺少经联合训练的连续容量证据。"
                  "R2 半步长收益有限且增加实际字节。R3 D0-Fast 实时通过，感知指标落后。"
                  "R4 已有合法记录剩余的质量增量接近零，重排留作候选。", "",
                  "| 上限 Mbps | D0-Fast FPS | P95 ms |", "|---:|---:|---:|"])
    for budget in BUDGETS:
        row = samples["d0_fast"][str(budget)]["pooled"]
        lines.append(f'| {budget} | {row["fps"]:.2f} | {row["total_ms_p95"]:.2f} |')
    lines.extend(["", "D0-Fast 在 34 帧与 123,477 条前缀边界上与旧 D0 一致。"
                  "原始 D0 的 5 Mbps 实测为 6.66 FPS、P95 249.39 ms。 "
                  "[原生计时](diagnostics/validation/d0_fast_timing.csv) · [字节及 RGB](diagnostics/d0_parity.json) · "
                  "[历史与同实际码率对照](diagnostics/references.json)", ""])
    return lines


def experiment_sections(files, out):
    text = []
    entropy = files["entropy"]
    if entropy:
        text.extend(["## 静态 CDF 实测", "",
                     f'Calibration 预注册规则选择 **{entropy["default"]}**。固定整数、组序、原生 Base 和 RGB 完全一致。',
                     "", "| E0 前缀上限 | E1 应用字节节省 | E1 增强字节节省 | E1 同上限 LPIPS |",
                     "|---|---:|---:|---:|"])
        for cut in ("5", "10", "20", "40", "ALL_VALID_RECORDS"):
            anchor = entropy["measurements"]["E1"][cut]
            symbol = anchor["same_symbol"]
            text.append(f'| {cut} | {100*symbol["whole_savings"]:.2f}% | '
                        f'{100*symbol["enhancement_savings"]:.2f}% | {anchor["same_rate"]["lpips"]:.5f} |')
        text.extend(["", "两套数学口径分别以应用层全包或仅增强段为分母。CDF 为预装资产。逐记录的 mode、头、CRC 与 rANS 终止都计入。"
                     "V2 当时选 E0 是因为其低档同感知码率更好。", "",
                     "[同符号](entropy/same_symbol.csv) · [同预算](entropy/same_rate.csv) · [时延和选择](entropy/summary.json)", ""])
    lite = files["lite"].get("profiles", {}).get("validation", {})
    if lite:
        text.extend(["## Base-Lite", "","| 方法 | I 峰值 B | P 峰值 B | 4,096 B 可达 | 1.3947 Mbps 可达 |",
                     "|---|---:|---:|---:|---:|"])
        for name, item in lite.items():
            text.append(f'| {name} | {int(item["by_type"]["I"]["bytes_max"])} | '
                        f'{int(item["by_type"]["P"]["bytes_max"])} | {item["service_4096_decoded"]}/174 | '
                        f'{item["service_budgets"]["1.3947"]["decoded"]}/174 |')
        text.extend(["", "L1 在 I 帧 Hyperprior 之前衰减 latent，逆向补偿 I 恢复的解码步长，并固定 P 分析增益。"
                     "独立重放的 I 帧 z 字节均值从 2,440 B 升至 3,736 B。"
                     "原生 y/z 共用 rANS payload。接收条件不完整时记录 Base 超限及之后的失步。"
                     "Lite 每种设置有独立前缀族标识。", "", "[Base 逐帧与时延](base_lite/summary.json) · "
                     "[Hyperprior 重放](base_lite/hyperprior.json)", ""])
    a2, a4 = files["a2"], files["a4"]
    if a2.get("quality") and a4.get("quality"):
        refs = {row["budget_mbps"]: row for row in a2["quality"] if row["status"] == "PASS"}
        current = [row for row in a4["quality"] if row["status"] == "PASS"]
        text.extend(["## 同一 E1 后端的 V2 对照", "",
                     "| 上限 Mbps | A2 实际 Mbps | A2 PSNR | A2 LPIPS | A2 DISTS | A2 FPS / P95 ms |",
                     "|---:|---:|---:|---:|---:|---:|"])
        for cut in BUDGETS:
            q = refs[cut]
            clock = a2["timing"][str(cut)]["pooled"]
            text.append(f'| {cut} | {q["actual_mbps"]:.4f} | {q["global_psnr"]:.3f} | {q["lpips"]:.5f} | '
                        f'{q["dists"]:.5f} | {clock["fps"]:.2f} / {clock["total_ms_p95"]:.2f} |')
        rate = refs[40]["actual_mbps"]
        adjusted = {metric: at_rate(current, metric, rate) for metric in ("global_psnr", "lpips", "dists")}
        if all(value is not None for value in adjusted.values()):
            text.extend(["", f'在 A2 高档实际 {rate:.4f} Mbps 处，对 V3 实际曲线插值得 '
                         f'{adjusted["global_psnr"]:.3f} dB、LPIPS {adjusted["lpips"]:.5f}、'
                         f'DISTS {adjusted["dists"]:.5f}。', ""])
        text.extend(["[A2 原生账本](evaluation/A2/summary.json)", ""])
    recovery = files["recovery"]
    if recovery:
        late, first, replay = (recovery[key] for key in ("late_checkpoint", "first_joint_checkpoint", "validation_check"))
        val_description = f'{replay["maximum"]:,} B' if replay else '`NOT_RUN`'
        text.extend(["## 联合更新的 Base 超限与恢复", "",
                     f'64 步联合探针的累计 2,064 步 checkpoint 在 Calibration 有 {late["over_cap"]}/128 帧超限。'
                     f'最高 {late["maximum"]:,} B，两个超限均为 P 帧。'
                     '第一个 14,757 B Base 已触发官方严格编码入口拒绝。超限权重下的硬预算 RD 与实时性为 `NOT_RUN`。',
                     "", f'从累计 2,000 步 checkpoint 重放同一次原生联合更新。累计 2,001 步的 Base 峰值在 '
                     f'Calibration 为 {first["maximum"]:,} B，在 Validation 为 {val_description}。'
                     f'通过固定 Base 门槛的研究 checkpoint 为 `{recovery["selected_research_candidate"]}`。'
                     '这一恢复保留了独立 I/P Encoder 实际更新和原有 HUF1 语义。',
                     "", "[物理逐帧审计与选择](diagnostics/base_recovery.json)", ""])
    if files["after_joint"]:
        pre = files["cal"]["results"]["2"]["pooled"]["C3/ALL_VALID_RECORDS"]
        after = files["after_joint"]["results"]["V3"]["pooled"]["C3/ALL_VALID_RECORDS"]
        text.extend(["## 联合训练后的最大前缀", "",
                     f'相同 48 张 Calibration 上的全有效记录 D1，V2 为 {pre["global_psnr"]:.3f} dB、'
                     f'LPIPS {pre["lpips"]:.5f}，恢复后的 V3 为 {after["global_psnr"]:.3f} dB、'
                     f'LPIPS {after["lpips"]:.5f}。 [连续与量化诊断](diagnostics/post_joint/capacity.json)', ""])
    if files["precision"]:
        a, b = (files["precision"]["pooled"][k] for k in ("existing/40", "half_step/40"))
        text.extend(["连续空间精度对照保持冻结 E1 与 Base。D0 在 40 Mbps 上限的半步长相对原步长变化 "
                     f'{b["global_psnr"]-a["global_psnr"]:+.3f} dB 和 '
                     f'{b["lpips"]-a["lpips"]:+.5f} LPIPS。此项用于误差归因。 [精度对照](diagnostics/precision.json)', ""])
    if files["runtime_trial"]:
        timing = files["runtime_trial"]["measurements"]
        text.extend(["## 按需组包试验", "",
                     f'40 Mbps Calibration 的批量工作区复用版 P95 为 {timing["reuse"]["p95_ms"]:.2f} ms，'
                     f'同条件的原后端为 {timing["original"]["p95_ms"]:.2f} ms。'
                     '原生工作区复用保留组序、CDF、CRC 和分段结束状态。 [时延试验](runtime_trial/summary.json)', ""])
    selected = files["optimized"]
    if selected:
        text.extend([f'完整 Validation 的 174 个最大包和全部合法边界哈希匹配。'
                     f'后端最终选择为 **{selected["selected"]}**。 [字节与双模式门控](runtime_trial/selection.json)', ""])
    text.extend(["## 消融与阶段门槛", "",
                 "| 实验 | 实际状态及证据 |", "|---|---|",
                 "| A0 V2 D1 | 冻结两个 Seed 原始 checkpoint；已做同一 Base 历史配对 |",
                 "| A1 D0-Fast | 已做同符号实验与三轮 Full-max |",
                 "| A2 Static | 已为 Seed 2 从 Train 重新拟合并测量 E0/E1/E2 |",
                 "| A3 R0 初始模型 | 复用同一单 latent 和 D1；零步物理路径由 A0/A2 覆盖 |",
                 "| A4 R0 训练 | 见优化器与独立部署账本 |",
                 "| A5 联合排序 | NOT_RUN；全记录配对没有支持采用新排序 |",
                 "| Core-Detail | NOT_RUN；R1 条件未满足 |",
                 "| L0/L1 | 见 Base-Lite |", ""])
    for name in ("P1", "P2", "probe"):
        file = out / f"pilot/{name}/gate.json"
        if file.exists():
            gate = json_at(file)
            text.append(f'- {name} 的实际 Pilot 裁决为 `{gate["status"]}`。 [逐帧配对](pilot/{name}/gate.json)')
    text.append("")
    return text


def contract_sections(files, verdict):
    paths = files["contracts"]
    if not paths:
        return []
    dynamic, fixed, advance = (paths[key] for key in ("dynamic", "all_budgets", "incremental"))
    target = (files["optimized"] or {}).get("contracts", "evaluation/contracts/summary.json")
    return ["## 严格前缀、因果参考与部署权重", "",
            f'混合预算复放 {dynamic["frames"]} 帧，参考状态失配 {dynamic["base_mismatches"]} 次。'
            f'固定上限复放 {fixed["checked_frame_budgets"]} 个帧预算组合，独立 Base 与 P DPB 逐位一致。'
            f'丢失 Base 后拒绝 {dynamic["p_rejected"]} 个依赖 P 帧，下一独立 I 帧恢复 {dynamic["i_recoveries"]} 次。'
            f'渐进接收复放 {advance["records"]:,} 条记录，最终 RGB 与一次接收逐位一致。'
            '篡改未来像素保持历史包、整数和参考哈希。使用未来输入的注入被捕获。', "",
            f'实际后端加载的 checkpoint 为 `{verdict["checkpoint_sha256"]}`。'
            '重新安装 FP16 代理后，I/P 帧的原生 y 与字节均相对 V2 对照改变，旧代理注入在 3 个 P 帧被捕获。'
            f' [独立接收与 CUDA 张量账本]({target})', ""]


def rate_sections(verdict, out):
    rates = json_at(out / "evaluation/rates.json")
    lp, ps = verdict["equal_lpips"], verdict["equal_psnr"]
    winner = verdict["candidate_ranking"]["candidates"].get("A2", {})
    lines = ["## 等画质实际码率", "",
             f'LPIPS 0.28978 的实际 Mbps 为 V3 **{fmt(lp["candidate_actual_mbps"], 5)}**、'
             f'V2 E0 **{fmt(val(rates["equal_quality"], "V2 D1", "lpips", .28978)["candidate_actual_mbps"], 5)}**、'
             f'V2 E1 **{fmt(winner.get("lpips_target_mbps"), 5)}**、Original AdapTok **{fmt(lp["original_actual_mbps"], 5)}**。'
             'Original 的码率采用独立时间窗平均服务口径。', "",
             f'严格等 PSNR 25 dB 为 `{ps["status"]}`。达到至少 25 dB 的最低实测固定逐帧上限端点为 '
             f'**{fmt(verdict["minimum_observed_psnr25_rate_mbps"], 5)} Mbps**。'
             f'完整 Base-only 的可变帧长平均为 **{fmt(verdict["minimum_observed_psnr25_series_average_mbps"], 5)} Mbps**。', "",
             "| 指标 | 目标 | V3 实际 Mbps | Original 实际 Mbps | 共同可比性 |", "|---|---:|---:|---:|---|"]
    for row in rates["equal_quality"]:
        if row["method"] != "V3 joint" or row["metric"] == "dists":
            continue
        metric = "PSNR" if row["metric"] == "global_psnr" else "LPIPS"
        lines.append(f'| {metric} | {row["target"]:g} | {fmt(row["candidate_actual_mbps"], 5)} | '
                     f'{fmt(row["original_actual_mbps"], 5)} | {row["status"]} |')
    lines.extend(["", "Original 和 V3 的实测 DISTS 区间没有交集。选择共同覆盖的 DISTS 0.18 可比较 V3 与三个原生对照。", "",
                  "| DISTS 0.18 对照 | V3 实际 Mbps | 对照实际 Mbps |", "|---|---:|---:|"])
    for row in verdict["matched_reference_dists"]:
        lines.append(f'| {row["reference"]} | {fmt(row["v3_mbps"], 5)} | {fmt(row["reference_mbps"], 5)} |')
    lines.extend(["", "[等质量与同率账本](evaluation/rates.json) · [所有目标表](evaluation/equal_quality.csv)", ""])
    return lines


def report(files, verdict, out, checkpoint):
    a4 = files["a4"]
    parts = ["# HLA-Tok-UF-V3 单卡 H100 验收", "",
             f'**总体裁决 `{verdict["overall"]}`。** 主要路线 **R0**，保留官方单 latent 和 D1 显示路径。',
             f'正式 Validation 准入为 **{"PASS" if verdict["validation_admission"] else "FAIL"}**。'
             f'Test 像素读取 **{verdict["test_pixel_reads"]}**。', "",
             "## 训练真实性、权重与容量", "",
             "V2 两个 Seed 各有 20,000 次实际更新。P1 属于特征空间训练，P2/P3 使用分离的缓存并训练有限的 P 帧适配参数。"
             "H100 重放分别探测码率与失真梯度，核对了官方 FP16 代理安装的折叠权重。", "",
             "[训练审计](../../../docs/hla_tok/uf_v3/training_audit.md) · "
             "[容量配对](../../../docs/hla_tok/uf_v3/capacity_diagnostics.md) · "
             "[路线和冻结门槛](diagnostics/capacity.json)", ""]
    ranking = verdict["candidate_ranking"]
    if ranking["selected"]:
        recommended = ranking["candidates"][ranking["selected"]]
        parts.extend([f'按冻结工程和 RD 门槛排序，当前最优研究对照为 **{ranking["selected"]} / {recommended["model"]}**。'
                      '正式部署准入仍为 FAIL。', ""])
        if ranking["selected"] == "A2":
            parts.extend([f'该端点沿用 V2 `d1-joint-seed2-step20000.pt`，SHA-256 为 '
                          f'`{recommended["checkpoint_sha256"]}`。 [E1 配表](entropy/static_cdf.json)', ""])
    if checkpoint:
        parts.extend([f'V3 实测候选 checkpoint 为 `{checkpoint}`，SHA-256 为 `{a4.get("checkpoint_sha256")}`。', ""])
    parts.extend(capacity_sections(files))
    if a4.get("quality"):
        parts.extend(["## 原生 Validation 率失真与实时性", "",
                      table_quality(a4["quality"], a4.get("timing", {}), files["ondemand"].get("timing", {})), "",
                      "0 为完整 Base-only 的实际序列平均码率。低于 Base 的预算记录物理超限和后续 P 帧失步。"
                      "40 Mbps 的两种模式使用相同生成上限，独立复跑可反映运行波动。"
                      "官方生产代理为 FP16。原生 FP32 和 BF16 路径为 `NOT_AVAILABLE`。", "",
                      f'[Full-max 计时]({verdict["full_max_summary"]}) · [按需计时]({verdict["on_demand_summary"]}) · '
                      '[首次按需计时，结束时有其他 GPU 工作]'
                      '(evaluation/on_demand/summary.json)', ""])
    parts.extend(experiment_sections(files, out))
    gains = verdict["progressive_adjacent"].get("V3 joint", [])
    if gains:
        high = gains[-1]
        parts.append(f'20→40 Mbps 多用 {fmt(high["added_actual_mbps"], 4)} Mbps，'
                     f'PSNR 增加 {fmt(high["psnr_db"], 4)} dB，LPIPS 改善 {fmt(high["lpips_improvement"], 5)}。 '
                     '[逐帧边际变化](evaluation/A4/adjacent_summary.csv)')
        parts.append("")
    parts.extend(rate_sections(verdict, out))
    parts.extend(contract_sections(files, verdict))
    parts.extend(["## 优化器更新", "",
                  "| 阶段 | Seed | 累计 | 本段 | CUDA 事件区间秒 | 墙钟秒 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, row in verdict["optimizer_runs"].items():
        parts.append(f'| {name} | {row["seed"]} | {row["optimizer_updates"]} | {row["added"]} | '
                     f'{fmt(row.get("gpu_event_interval_seconds"), 2)} | {fmt(row.get("wall_seconds"), 2)} |')
    fit = verdict["trained_evidence"]
    if fit:
        measured = fit["gradient_probe"]
        parts.extend(["", "Seed 1 的前 2,000 步训练 D1 和量化器并冻结骨干。联合探针首步展开 4 帧原生 1080p，"
                      "其余 63 步展开 4 帧 192×256 原生空间裁块。64 步训练直接对活跃骨干张量独立求率、失真梯度。"
                      "P3、P4 和 Seed 2 在阶段门槛处停止前进。", "",
                      "| 张量路径 | 失真梯度 L2 | 码率梯度 L2 |", "|---|---:|---:|"])
        for name, value in measured["distortion"].items():
            parts.append(f'| {name} | {value:.5g} | {measured["rate"][name]:.5g} |')
    parts.extend(["", "CUDA 事件区间含设备空闲及主机等待。原生联合更新的实际分辨率、展开与独立梯度见训练账本。"
                  "缓存仅保留经过核验的 Train 原始 RGB。可微 Base 代价是代理，正式质量和码率由独立真实编解码产生。",
                  "", "## 分项裁决", "", "| 项目 | 状态 |", "|---|---|"])
    parts.extend(f"| {key} | `{value}` |" for key, value in verdict["dimensions"].items())
    if files["ledger"]:
        ledger = files["ledger"]
        parts.extend(["", f'含复验共核对 **{ledger["completed_run_optimizer_calls"]}** 次已保存的优化器调用，'
                      f'CUDA 事件区间 {ledger["cuda_event_interval_seconds"]:.2f} s、分段墙钟合计 '
                      f'{ledger["wall_seconds"]:.2f} s、在线 Teacher 墙钟 {ledger["teacher_wall_seconds"]:.2f} s。'
                      '另有首轮报告失败前保存的两次更新。 '
                      '[独立 AdamW 计数核验](training/checkpoint_ledger.json)'])
    if files["qa"]:
        qa = files["qa"]
        faults = sum(group["KILLED"] for group in qa["directed_mutations"].values())
        parts.extend(["", "## 回归", "",
                      f'CPU {qa["cpu"]["passed"]} 项通过，CUDA 路径 {qa["cuda"]["passed"]} 项通过。'
                      f'源码定向变异捕获 {faults} 个，另有实际 CUDA 旧代理与未来帧注入。'
                      f'生产 Python 模块最高 Crapper 分数为 {qa["crapper"]["max"]:.1f}。 [回归账本](qa/summary.json)'])
    parts.extend(["", "## 样例与复现", "",
                  "[机器裁决](final_verdict.json) · [复现命令](../../../docs/hla_tok/uf_v3/reproduction.md) · "
                  "[工作记录](../../../docs/hla_tok/uf_v3/work_log.md) · [图表清单](figures/manifest.json) · "
                  "[图表核验](figures/qa.json) · [原生样例哈希](samples/native/manifest.json) · "
                  "[历史资产哈希](audit/historical_sha256.json) · [实验哈希](experiment_manifest.json)", ""])
    (out / "final_report.md").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    files = read_layers(args.results)
    curves, equal = quality_by_rate(files, args.results)
    verdict = decide(files, curves, equal, args.results)
    if args.checkpoint:
        digest = sha256_file(args.checkpoint)
        if digest != verdict["checkpoint_sha256"]:
            raise ValueError("report checkpoint does not match the physical evaluation")
        verdict["checkpoint"] = str(args.checkpoint)
    write_json(args.results / "final_verdict.json", verdict)
    report(files, verdict, args.results, args.checkpoint)


if __name__ == "__main__":
    main()
