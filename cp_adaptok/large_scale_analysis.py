"""Post-training entropy/scorer fitting, sealed final tests, and honest experiment reports."""

import csv
import json

import numpy as np
import torch

from cp_adaptok.large_scale import gpu_snapshot, memory_fraction
from cp_adaptok.large_scale_data import file_hash, video_specs
from cp_adaptok.large_scale_tracking import TrainingObserver, read_json
from cp_adaptok.large_scale_validation import validate_checkpoint
from cp_adaptok.runtime import (
    MissingPrerequisites,
    build_model,
    load_checkpoint,
    read_config,
    resolve,
    seed_everything,
    write_json,
)
from cp_adaptok.training import run_training
from cp_adaptok.training_state import stable_hash

RD_FIELDS = ["run_id", "branch", "phase", "split", "tokens", "psnr", "lpips", "dists", "ms-ssim", "bpp",
             "i_psnr", "p_psnr", "model_id", "entropy_id"]


def fit_auxiliary(manifest, splits, branch):
    """Stage3 is retrained for this VQ, then calibrated independently of held-out content."""
    root = resolve("results/large_scale") / manifest["run_id"]
    config = read_config(root / "s4" / branch / "config.json")
    source = resolve(config["paths"]["save_checkpoint"]).with_name("best_validation.pt")
    output = root / "stage3" / branch
    target = resolve(config["large_scale"]["checkpoint_root"]) / manifest["run_id"] / "stage3" / branch / "latest.pt"
    count = config["large_scale"]["auxiliary_updates"]
    config.update(stage=3, teacher={"policy": "none"})
    train = config["training"]
    train.update(crop=512, sequence_length=4, sequence_schedule=[], gradient_accumulation=1, lr=0.001,
                 save_every=250, validate_every=count, full_validate_every=count,
                 scheduler={"warmup_steps": 50, "total_steps": count, "min_lr_ratio": 0.1})
    train["vq"].update(strategy="deterministic_nearest", warmup_steps=0, initialization=None)
    train["all_budgets"] = True
    config["prefix_training"] = {"mode": "legacy", "budgets": train["budgets"]}
    config["loss"].update(latent=0, lpips=0, dists=0, estimated_rate=0)
    config["data"]["train"] = video_specs(splits["train"], crop=512)
    config["paths"].update(checkpoint=str(source), output=str(output), save_checkpoint=str(target))
    config["experiment"].update(phase="stage3", target_updates=count, initial_checkpoint_sha256=file_hash(source))
    config["experiment"].pop("config_hash", None)
    config["experiment"]["config_hash"] = stable_hash(config)
    previous = output / "config.json"
    if previous.exists() and stable_hash(read_json(previous, {})) != stable_hash(config):
        raise ValueError("Stage3 inputs changed; refuse to reuse previous auxiliary weights")
    write_json(previous, config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, artifact = build_model(config, config["device"], checkpoint_override=str(target) if target.exists() else None)
    observer = TrainingObserver(config, splits, manifest)
    try:
        run_training(model, config, count, artifact, target.exists(), "trained", observer=observer)
        artifact = load_checkpoint(target)
        observer._phase_state(artifact["step"], artifact["training_state"], "complete", target)
        output_rd = output / "development_calibrated"
        report = validate_checkpoint(model, config, splits, "full", output_rd, target)
        return config, output_rd / "calibrated.pt", report
    finally:
        del model
        torch.cuda.empty_cache()


def evaluate_experiment(manifest, splits):
    if any(manifest["phases"].get(f"s4/{branch}", {}).get("status") != "complete" for branch in ("B0", "B1")):
        raise MissingPrerequisites(["both B0 and B1 must finish S1-S4 before final model selection and testing"])
    settings = read_config("configs/large_scale/common.yaml")["large_scale"]
    memory_fraction(gpu_snapshot(), settings)
    root = resolve(settings["output_root"])
    selected = {}
    for branch in ("B0", "B1"):
        config, path, development = fit_auxiliary(manifest, splits, branch)
        selected[branch] = {"checkpoint": str(path), "sha256": file_hash(path),
                            "model_id": development["model_id"], "entropy_id": development["entropy_id"],
                            "config": config}
    seal_path = root / manifest["run_id"] / "final_selection.json"
    seal = {"run_id": manifest["run_id"], "selected": selected,
            "test_manifest_hash": manifest["manifest_hashes"]["test"],
            "policy": "fixed before final test; final results must not tune this experiment"}
    if seal_path.exists() and read_json(seal_path, {}) != seal:
        raise ValueError("final selection is already sealed; cannot change it after reading final test")
    write_json(seal_path, seal)
    manifest["final_test_frozen"] = True
    reports = {}
    for branch, selected_model in selected.items():
        output = root / manifest["run_id"] / "final_test" / branch
        if (output / "validation.json").exists():
            report = read_json(output / "validation.json", {})
            if report["model_id"] != selected_model["model_id"]:
                raise ValueError("saved final test has a different sealed model")
        else:
            config = selected_model["config"]
            seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
            model, _, _ = build_model(config, config["device"], checkpoint_override=selected_model["checkpoint"])
            try:
                report = validate_checkpoint(model, config, splits, "final", output, selected_model["checkpoint"])
            finally:
                del model
                torch.cuda.empty_cache()
        if report["model_id"] != selected_model["model_id"] or report["entropy_id"] != selected_model["entropy_id"]:
            raise ValueError("final stream model/probability identity differs from the sealed artifact")
        reports[branch] = report
    comparison = compare_reports(reports["B0"], reports["B1"])
    write_json(root / manifest["run_id"] / "comparison.json", comparison)
    write_rd_table(root / "rd_comparison.csv", manifest["run_id"], reports)
    manifest["final_test_complete"] = True
    write_json(root / manifest["run_id"] / "experiment_manifest.json", manifest)
    write_json(root / "experiment_manifest.json", manifest)
    return build_report(manifest)


def compare_reports(baseline, nested, seed=47, bootstrap_samples=2000):
    def frames(report):
        return {(r["source"]["content_id"], r["frame_id"]): r for r in report["records"]}

    a, b = frames(baseline), frames(nested)
    if set(a) != set(b):
        raise ValueError("B0/B1 final evaluations must use identical held-out frames")
    grouped, paired, rd = {}, [], []
    for key in a:
        left, right = a[key]["prefixes"], b[key]["prefixes"]
        for x, y in zip(left, right):
            delta = y["psnr"] - x["psnr"]
            grouped.setdefault((key[0], x["tokens"]), []).append(delta)
            paired.append({"content_id": key[0], "frame": key[1], "tokens": x["tokens"], "delta_psnr": delta})
        rd.extend(same_rate_points(key, left, right))
    rng = np.random.default_rng(seed)
    summary = []
    for budget in (128, 256, 512, 1024, 2048):
        means = np.array([np.mean(v) for (content, k), v in grouped.items() if k == budget])
        ci = np.percentile(rng.choice(means, (bootstrap_samples, len(means)), replace=True).mean(1), [2.5, 97.5])
        summary.append({"tokens": budget, "mean_video_delta_psnr": float(means.mean()), "ci95": ci.tolist()})
    return {"paired_frames": paired, "sequence_bootstrap": summary, "same_bpp_interpolation": rd,
            "bootstrap_unit": "independent video, not frames or training seeds", "single_training_seed": True,
            "equal_flops_claim": False, "same_compute_caveat": "decoder calls and GPU timeline are proxies only"}


def same_rate_points(key, left, right):
    def curve(rows):
        ordered = sorted(rows, key=lambda r: r["bpp"])
        return np.array([r["bpp"] for r in ordered]), np.array([r["psnr"] for r in ordered])

    x, y = curve(left)
    u, v = curve(right)
    low, high = max(x.min(), u.min()), min(x.max(), u.max())
    if high < low or low <= 0:
        return []
    rates = np.geomspace(low, high, 5)
    deltas = np.interp(np.log(rates), np.log(u), v) - np.interp(np.log(rates), np.log(x), y)
    return [{"content_id": key[0], "frame": key[1], "bpp": float(rate), "delta_psnr": float(delta),
             "method": "interpolated PSNR on overlapping log measured-bpp range; no extrapolation"}
            for rate, delta in zip(rates, deltas)]


def write_rd_table(path, run_id, reports):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for branch, report in reports.items():
            types = report["budget_summary_by_frame_type"]
            for i, row in enumerate(report["budget_summary"]):
                writer.writerow({"run_id": run_id, "branch": branch, "phase": "final", "split": "test", **row,
                                 "i_psnr": types["I"][i]["psnr"], "p_psnr": types["P"][i]["psnr"],
                                 "model_id": report["model_id"], "entropy_id": report["entropy_id"]})


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot_curves(root, run_id):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / run_id / "figures"
    output.mkdir(parents=True, exist_ok=True)
    train = [r for r in csv_rows(root / "training_metrics.csv") if r["run_id"] == run_id]
    validation = [r for r in csv_rows(root / "validation_metrics.csv") if r["run_id"] == run_id]
    plots = []
    tasks = [(train, "loss", "Train objective"), (train, "vq_perplexity", "Training VQ perplexity"),
             (train, "frames_per_s", "Training frames / s"), (train, "peak_allocated_mib", "Peak GPU MiB"),
             (validation, "psnr", "Held-out PSNR (dB)"), (validation, "lpips", "Held-out LPIPS"),
             (validation, "dists", "Held-out DISTS"), (validation, "i_psnr", "I-frame PSNR (dB)"),
             (validation, "p_psnr", "P-frame PSNR (dB)"),
             (validation, "deterministic_perplexity", "Deterministic VQ perplexity")]
    for rows, metric, label in tasks:
        groups = curve_groups(rows, metric)
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, values in groups.items():
            ax.plot([x for x, y in values], [y for x, y in values], label=name, linewidth=1)
        ax.set(xlabel="Optimizer updates in phase", ylabel=label)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(str(path))
    return plots


def curve_groups(rows, metric):
    groups = {}
    for row in rows:
        if not row.get(metric):
            continue
        suffix = f" K{row['tokens']} {row['tier']}" if "tokens" in row else ""
        name = f"{row['phase']}/{row['branch']}{suffix}"
        groups.setdefault(name, []).append((int(row["step"]), float(row[metric])))
    return groups


def build_report(manifest):
    root = resolve("results/large_scale")
    summary = {"run_id": manifest["run_id"], "git_sha": subprocess_sha(), "dataset": manifest["dataset_summary"],
               "manifest_hashes": manifest["manifest_hashes"], "phases": manifest["phases"],
               "final_test_complete": manifest.get("final_test_complete", False),
               "convergence_verified": False, "single_training_seed": True, "equal_flops_claim": False,
               "figures": plot_curves(root, manifest["run_id"]),
               "interpretation": "Actual completed updates only; a pilot is not sufficient large-scale training."}
    if not (root / "rd_comparison.csv").exists():
        write_rd_table(root / "rd_comparison.csv", manifest["run_id"], {})
    write_json(root / "final_summary.json", summary)
    write_json(root / manifest["run_id"] / "summary.json", summary)
    write_report_markdown(summary, root)
    return summary


def subprocess_sha():
    import subprocess

    return subprocess.check_output(["git", "-C", str(resolve(".")), "rev-parse", "HEAD"], text=True).strip()


def write_report_markdown(summary, root):
    rows = []
    for name, phase in summary["phases"].items():
        stats = phase["statistics"]
        rows.append(f"| {name} | {phase['status']} | {phase['completed_updates']}/{phase['target_updates']} | "
                    f"{stats.get('frames', 0)} | {stats.get('training_wall_s', 0):.1f} | "
                    f"{stats.get('distinct_videos', 0)} | {stats.get('distinct_windows', 0)} |")
    text = f"""# CP-AdapTok 大规模训练执行报告

Git SHA：`{summary['git_sha']}`。实验：`{summary['run_id']}`。日期和日志时区：Asia/Shanghai。

只报告真实完成的更新；`pilot` 是有限预实验。最终测试完成：`{summary['final_test_complete']}`；
收敛已验证：`{summary['convergence_verified']}`；当前仅单训练种子，不能据此宣称统计稳定性。

## 数据与隔离

```json
{json.dumps(summary['dataset'], indent=2, ensure_ascii=False)}
```

清单逐帧审计，记录原始分辨率、全部帧数、FPS 及名义标记、来源、官方 split、内容 ID、SHA256 和裁剪支持。
训练只允许官方 Train。历史 MCL-JCV、UVG、Basketball 和此前 DAVIS Val 开发记录保留，禁止训练。
最终测试清单在训练前封存，只有 S4 和新模型 Stage3 完成并固定模型身份后才能执行。
数据补充来源、CRC、许可及内容划分见 `dataset_audit.json` 和本实验 `manifests/`。

## 实际训练

| 阶段/分支 | 状态 | 更新数/目标 | 处理帧数 | 更新墙钟秒 | 不同视频 | 不同窗口 |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

各阶段状态、Checkpoint、上游版本、配置与模型结构哈希、验证记录见 `experiment_manifest.json`；
所有优化器、Scheduler、四个 RNG 及累计训练状态保存到 `latest.pt`。
原始逐帧训练日志为各阶段 `train.jsonl`，实际窗口和增强为 `samples.jsonl`。

## 固定计划与公平性

- S0：2000 更新，512 crop，4 帧、TBPTT 2、累积 2、AdamW 3e-4；前 200 次连续表示，随后仅用 Train Pre-VQ 特征做一次 K-means。随机 VQ 温度 0.1 → 0.01，Teacher 仅 I 帧且逐步衰减。
- S1：每分支 8000 更新，512 crop，4 帧、TBPTT 2、累积 2、LR 1e-4。
- S2：每分支 12000 更新，原生 720p crop，4 帧、TBPTT 2、累积 2、LR 5e-5。
- S3：每分支 8000 更新，原生 1080p crop，4 帧、TBPTT 2、累积 4、LR 2e-5。
- S4：每分支 4000 更新，原生 1080p，8 → 16 连续帧、TBPTT 2、累积 4、LR 1e-5。
- 全部阶段采用非零起点线性 Warmup + Cosine，最低 LR 为起始的 0.1；S0 Warmup 100，其余正式阶段 200。Pilot 独立使用 200 更新、Warmup 20、20 次连续表示，用于覆盖 VQ 切换，不能充当正式 S0。
- B0 保留原多预算采样；B1 每次包含 128、2048 和随机中间预算，启用原 Nested 单向约束。结构、骨干、共享 S0、数据顺序、增强、基础损失、优化器和计划保持相同。
- 同更新次数不代表同 FLOPs。逐预算监督次数、完整 Decoder 调用数、处理帧数、GPU 时间线耗时、墙钟时间均独立记录。GPU 时间线包含主机造成的间隙，不能冒充内核活跃时间；Decoder 次数也不能冒充实测 FLOPs。
- BF16、Grad Clip 1.0；不再次放大迁移 Query，不再次 K-means，不沿用上一阶段动量。默认 Rate Loss 为 0，避免未校准的率项造成塌缩；LPIPS 由 S1 0.01 到 S2–S4 0.03，DISTS 只评估。尚未执行的强率失真微调不作完成声明。

## 验证、模型选择与真实码率

主要预算 `[128,256,512,1024,2048]`。快速验证使用独立视频和确定性离散 VQ；
完整开发验证与最终测试使用真实 rANS、Frame Header、Packet 和传输 framing，校验发送/接收 Base-only 状态一致。
每次完整评估先做 Train-only 熵校准，单独保存带概率表身份的 `calibrated.pt`，不得恢复表示优化器。
最终模型重新训练 Stage3 熵模型和 Scorer；旧 Scorer 不用于正式性能结论。

预先固定选择公式：所有预算平均 MSE + 0.25×(K128 MSE + K2048 MSE)
+ 0.001×质量倒退率 + 0.001×平均 LPIPS + 0.001×平均 DISTS + 0.0001×平均真实 bpp。
只比较相同层级和相同样本的验证分数，Full 层级建立后优先 Full；快速验证缺失的指标不填假值。
不以 DeltaPSNR 或 Perplexity 单独选模。不根据单个验证点或早期前缀收益停止；自动平台早停关闭。

各预算 PSNR/LPIPS/DISTS、I/P PSNR：`validation_metrics.csv`；真实 RD：`rd_comparison.csv`；
最终 B0/B1 逐帧与视频 Bootstrap：本实验 `comparison.json`（仅在实际完成后生成）。
未产生的数值表示尚未测量，不能解释为零。只有完成封存测试后才讨论同 bpp 插值比较。

## 复现与继续

```bash
.venv-native/bin/python -m scripts.prepare_large_scale_data
.venv-native/bin/python -m scripts.prepare_reds_large_scale --sequences 60
.venv-native/bin/python -m scripts.train_large_scale --phase preflight
.venv-native/bin/python -m scripts.train_large_scale --phase pilot --run-id pilot_20261116
.venv-native/bin/python -m scripts.train_large_scale --run-all --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --resume --run-all --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --phase s2 --branch both --run-id formal_20261116
.venv-native/bin/python -m scripts.train_large_scale --phase evaluate --run-id formal_20261116
.venv-native/bin/python -m scripts.report_large_scale --run-id {summary['run_id']}
```

`--resume` 恢复同阶段绝对目标，不追加额外预算。输入清单和配置变化会拒绝恢复；
数据扩充后使用新 run-id，以免改变已有实验的数据定义。
持续过程中的状态以 `state.json` 和 `latest.pt` 为准，汇总报告为生成时快照。

## 验收与研究边界

测试日志：`results/large_scale/logs/pytest.txt`、`resume_acceptance.txt`；ub-mode 行为与决策日志：
`docs/large_scale_work_log.md`。图表保存在本实验 `figures/`，只有实际存在的数据才绘图。
仍需依据充分训练后的独立验证判断绝对画质、后续 Token 增益、I/P 差异、过拟合、VQ 二次塌缩和验证平台。
资源或数据阻塞记录在 `blocked.json` / `interruption.json`；未完成预算保留，不编造收敛或最终 RD。
"""
    resolve("docs/large_scale_training_report.md").write_text(text)
