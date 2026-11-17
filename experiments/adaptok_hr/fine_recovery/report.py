"""Render research decisions from measured JSON; never substitute planned results."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from adaptok_hr.runtime import environment, file_hash, write_json

ROOT = Path("results/adaptok_hr_fine_recovery")
EVIDENCE = Path("experiments/adaptok_hr/fine_recovery/evidence")


def read(path):
    return json.loads(Path(path).read_text())


def validation_results():
    return [(path, read(path)) for path in sorted(ROOT.glob("A*/validation_*f/summary.json"))]


def screen():
    records = {name: read(ROOT / name / "validation_000300_8f/summary.json") for name in ("A0", "A1", "A2", "A3")}
    def key(name):
        r = records[name]
        c, q = r["criteria"], r["closed_loop"]["2048"]
        effective = c["fine_multiple_ids"] and c["fine_measurable_contribution"]
        return (-int(effective), -c["same_context_gain_db"], q["lpips"], -q["psnr_global"])
    names = sorted(records, key=key)
    write_json(ROOT / "selection_300.json", {"selected": names[:2], "ranking": names,
               "steps_each": 300, "frames_each": 1200, "validation_frames_per_source": 8,
               "rule": read("experiments/adaptok_hr/fine_recovery/protocol.json")["selection_order"],
               "criteria": {n: records[n]["criteria"] for n in names}})
    print(json.dumps({"selected": names[:2], "criteria": {n: records[n]["criteria"] for n in names}}))


def experiment_table(records):
    rows = ["| 配置 | 更新/帧次 | 验证帧 | Fine 码字 / perplexity / 最大占比 | 同状态 ΔPSNR | 闭环 PSNR | LPIPS | DISTS | bpp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path, r in records:
        name = path.parents[1].name
        p, c, q = r["protocol"], r["criteria"], r["closed_loop"]["2048"]
        fine = r["same_context"]["2048"]["levels"]["Fine"]
        rows.append(f"| {name} | {p['completed_steps']}/{p['training_frames']} | {q['frames']} | "
                    f"{fine['used_codes']} / {fine['perplexity']:.3f} / {fine['maximum_code_fraction']:.4%} | "
                    f"{c['same_context_gain_db']:+.5f} | {q['psnr_global']:.5f} | {q['lpips']:.6f} | {q['dists']:.6f} | {q['actual_bpp']:.8f} |")
    return rows


def final_decision():
    selected = read(ROOT / "selection_300.json")["selected"]
    records = {n: read(ROOT / n / "validation_001000_64f/summary.json") for n in selected}
    info = {n: r["criteria"]["fine_multiple_ids"] and r["criteria"]["fine_measurable_contribution"] for n, r in records.items()}
    eligible = [n for n in selected if info[n]]
    decision = {"selected_300": selected, "fine_information_at_1000": info,
                "decoder_experiments_allowed": bool(eligible), "decoder_candidate_names": eligible,
                "scaling_or_performance_optimization_allowed": False,
                "reason": "Fine information is a prerequisite; all other quality, late-frame and speed gates must also pass before scaling.",
                "negative_result_stop": not bool(eligible),
                "new_final_test_run": False, "historical_final_test_used_for_selection": False}
    write_json(ROOT / "decision.json", decision)
    print(json.dumps(decision))
    return decision


def archive():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in sorted(ROOT.rglob("*")):
        if path.suffix in (".json", ".jsonl", ".xml", ".md", ".bin") or (
                path.suffix == ".png" and "car-roundabout_0000" in path.name):
            target = EVIDENCE / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in sorted(Path("checkpoints/adaptok_hr_fine_recovery").rglob("*.json")):
        target = EVIDENCE / "checkpoint_metadata" / path.relative_to("checkpoints/adaptok_hr_fine_recovery")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in sorted((ROOT / "figures").glob("*")):
        if path.is_file():
            target = EVIDENCE / "figures" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def write_report():
    records = validation_results()
    e0 = read(ROOT / "E0/summary.json")
    rows = ["# AdapTok-HR Fine Token 恢复实验报告", "",
            "本报告由实际 JSON/JSONL 自动生成。尚未执行的阶段明确列为未执行；单种子 Pilot 不代表训练收敛。", "",
            "## 实验身份与保护", "", f"基线 `cc87bd4`；分支 `experiment/adaptok-hr-fine-recovery`。数据身份 `{e0['manifest_identity']}`。",
            f"旧 1000 步 Checkpoint SHA256：`{e0['checkpoint_sha256']}`。Train/Validation 原生 RGB 不缩放；未以旧 Test 或历史视频选择模型。", "",
             "## E0：冻结模型的确定事实", "", (ROOT / "E0/root_cause.md").read_text().replace("# E0 冻结权重诊断", ""),
            "", "实际 STE 与 Decoder 恢复图梯度补充见 `evidence/E0/actual_ste_gradients.json`；连续旁路与真实离散前向梯度分开记录。", "",
            "## E1/E2：实现与匹配条件", "",
            "统一球面 VQ 产生逐 Token Codebook/Commitment Loss；active_prefix 对每个重建预算分别取前缀均值，再按重建的 1/2、1/2 权重混合；all_candidates 保留全候选对照。前向仍为真实码本 Embedding，STE 只改变反向。",
            "保持旧损失的逐元素 MSE 尺度：每 Token 16 维先取均值，再归约 Token；没有同时改变 VQ 权重或增加均衡正则。训练行的 legacy quantization_mse 字段仍表示全部候选；实际发送的逐层误差和方差只读取 levels，完整诊断按 all_candidates/active_K* 分开，绝不混用。",
            "分层模式拥有三个独立 8192×16 码本，接收端依确定性位置表查表；包仍为 ATHR v1、13-bit ID、142-byte 帧头，无逐 Token 层级字段。哈希包含全部真实码本和配置，因此错误模型或旧包会被拒绝，没有身份兼容回退。",
            "模型初始公共权重、数据计划和主/辅助预算序列逐条核对，证据为 matching_*.json。共同 seed=20261118、数据 seed=20261119、预算 seed=20261120；Batch=1、四帧展开、TBPTT2、AdamW 1e-4。除两个消融因素外训练权重一致，保留原单调性权重 0.1。", "",
            "## E3：匹配 2×2 的实测结果", "",
            "300 步筛选：每来源前 8 帧，共 24 帧；1000 步完整固定验证：每来源最多 64 帧，共 174 帧（blackswan 原片只有 50 帧、dog 60 帧、car-roundabout 64 帧）。这三个来源沿用旧实验预先固定的 Validation 子集，不是全部 21 个 Validation 来源。不同验证长度不直接作效果排名。",
            "同状态 ΔPSNR = 当前完整 K2048 码流历史下 K2048−K1024；闭环指标为各预算各自推进真实码流。Fine 用量仅统计 K2048 实际发送的位置。", ""]
    rows += experiment_table(records)
    if (ROOT / "selection_300.json").exists():
        selected = read(ROOT / "selection_300.json")["selected"]
        rows += ["", f"按预先固定规则续训：{', '.join(selected)}。其余两组保留 300 步结果，不拿 300 步与 1000 步跨步数作结构优劣结论。"]
    if (ROOT / "decision.json").exists():
        d = read(ROOT / "decision.json")
        rows += ["", "## 后续门槛", "", f"1000 步 Fine 信息门槛：`{json.dumps(d['fine_information_at_1000'])}`。"]
        if d["negative_result_stop"]:
            rows.insert(2, "结论：预算感知 VQ 与分层码本的受控对照未恢复 Fine 有效信息；按 1000 步停止条件保留负结果，没有扩大训练或增加 Decoder。\n")
            rows += ["Fine 有效传递仍未恢复。本轮保留负结果并停止扩展训练。D1 Fine Residual、D2 Detail Loss、D3 二者组合均未执行；未新增 Decoder 参数或高频损失。未开展更大数据/16–32 帧正式训练或 30FPS 优化。D0 为四组实际使用的原 Decoder。",
                     "没有新的授权外部视频；复用原 manifest 导入接口和隔离规则，没有自动下载数据。新来源最终封存测试尚未建立、未执行。"]
    rows += ["", "## 连续 64 帧吞吐", "",
             "下面均为 car-roundabout 原生连续 64 帧，独立收发端模型实例；包含发送端本地解码、打包、接收端解码。无感知网络常驻；排除读盘/H2D/指标/写盘/断言，不是 Encoder 单前向吞吐。", "",
             "| Checkpoint | K | Encoder FPS | Sender FPS | Receiver FPS | 顺序联合 FPS | 联合 P50/P95 ms | 峰值 MiB | 参数 | bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path in sorted(ROOT.glob("*/benchmark_*.json")):
        r = read(path)
        for k, b in r["budgets"].items():
            t = b["timing"]
            rows.append(f"| {path.parent.name}/{r['completed_steps']} | {k} | {t['encoder']['fps']:.2f} | {t['sender']['fps']:.2f} | {t['receiver']['fps']:.2f} | {t['joint']['fps']:.2f} | {t['joint']['p50_ms']:.3f}/{t['joint']['p95_ms']:.3f} | {b['peak_gpu_mib']:.1f} | {r['parameters']} | {b['bytes']} |")
    rows += ["", "## 测试、可复现性与交付", "",
             "原始 JSON/JSONL、Checkpoint 元数据、环境与代码身份归档到 experiments/adaptok_hr/fine_recovery/evidence；权重保存在独立 checkpoints/adaptok_hr_fine_recovery 目录。运行命令见同目录 README。",
             "测试 XML、逐位恢复、23 类行为变异的初次结果及修复后证据均保留。没有把第一次存活变异隐藏为首次通过。性能计时与覆盖率/Profiler 分离。",
             "单种子、固定小规模 Train/Validation Pilot 的结果只说明该设置下是否恢复 Fine；不能证明更大数据或充分收敛后的上限。统计无独立种子置信区间，没有伪造误差条。", "",
             "## 原始数据图", ""]
    for path in sorted((ROOT / "figures").glob("*.png")):
        rows.append(f"![{path.stem}](../experiments/adaptok_hr/fine_recovery/evidence/figures/{path.name})")
    rows += ["", "## 实际 1080p 重建示例", "",
             "以下为同一原始 car-roundabout 第 0 帧与各预算真实 ID 解码，全部保留 1920×1080，未进行锐化或后处理。", ""]
    for path in sorted(ROOT.glob("A*/validation_001000_64f/samples/*car-roundabout_0000*.png")):
        relative = path.relative_to(ROOT)
        rows.append(f"[{path.parents[2].name}: {path.name}](../experiments/adaptok_hr/fine_recovery/evidence/{relative})  ")
    Path("docs/adaptok_hr_fine_recovery_report.md").write_text("\n".join(rows) + "\n")
    archive()
    write_json(EVIDENCE / "evidence_manifest.json", {"environment": environment(), "files": {
        str(p.relative_to(EVIDENCE)): file_hash(p) for p in sorted(EVIDENCE.rglob("*")) if p.is_file()
        and p.name != "evidence_manifest.json"}, "commits": subprocess.check_output(
            ["git", "log", "--oneline", "cc87bd4..HEAD"], text=True).splitlines()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "decide", "report", "archive"))
    mode = parser.parse_args().mode
    {"screen": screen, "decide": final_decision, "report": write_report, "archive": archive}[mode]()


if __name__ == "__main__":
    main()
