"""Report the actual pretrained/native follow-up without replacing historical smoke results."""

import csv
import json
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/pretrained_native"


def read(path):
    return json.loads((ROOT / path).read_text())


def values(rows, key):
    return [row[key] for row in rows]


def training_updates():
    return [
        json.loads(line) for line in (ROOT / "results/pretrained_s1/train.jsonl").read_text().splitlines()
    ][:100]


def perplexities(updates):
    return [mean(values(update["records"], "vq_perplexity")) for update in updates]


def maximum_cache_error(records):
    return max(row["cache_max_error"] for frame in records for row in frame["prefixes"])


def figures(initial, trained, native, updates):
    destination = OUTPUT / "figures"
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    native_bpp = mean(values(native["records"], "total_bytes")) * 8 / (1920 * 1080)
    for axis, metric in zip(axes[:2], ("psnr", "lpips")):
        for report, label in ((initial, "Before CP training"), (trained, "100 CP updates")):
            rows = report["budget_summary"]
            axis.plot(values(rows, "bpp"), values(rows, metric), "o-", label=label)
        axis.plot(native_bpp, mean(values(native["records"], metric)), "ks", label="Native QP32")
        axis.set(xlabel="Measured bpp (see framing notes)", ylabel=metric.upper())
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[2].semilogy(values(updates, "step"), perplexities(updates))
    axes[2].set(xlabel="Training update", ylabel="Mean VQ perplexity", ylim=(0.9, 20))
    axes[2].grid(alpha=0.2)
    fig.suptitle("Short engineering run: VQ collapse; compression quality target unmet")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(destination / f"pretrained_quality.{extension}", dpi=160)
    plt.close(fig)


def budget_table(initial, trained):
    rows, table = (
        [],
        ["| K | 初始 PSNR | 100步 PSNR | LPIPS | DISTS | 实际 bpp |", "|---:|---:|---:|---:|---:|---:|"],
    )
    for before, after in zip(initial["budget_summary"], trained["budget_summary"]):
        rows.append(
            dict(
                tokens=after["tokens"],
                initial_psnr=before["psnr"],
                initial_lpips=before["lpips"],
                **{key: after[key] for key in ("psnr", "ms-ssim", "lpips", "dists", "bpp", "kbps")},
            )
        )
        table.append(
            f"| {after['tokens']} | {before['psnr']:.4f} | {after['psnr']:.4f} | {after['lpips']:.4f} | {after['dists']:.4f} | {after['bpp']:.8f} |"
        )
    with (OUTPUT / "budget_quality.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, table


def main():
    initial = read("results/pretrained_initial/eval/evaluation.json")
    trained = read("results/pretrained_s1_test/eval/evaluation.json")
    native = read("results/pretrained_native_baseline/eval/native_baseline.json")
    long = read("results/pretrained_long_eval/eval/evaluation.json")
    two = read("results/pretrained_benchmark_two/benchmark.json")
    five = read("results/pretrained_benchmark_five/benchmark.json")
    native_speed = read("results/pretrained_native_benchmark/benchmark.json")
    tcp = read("results/pretrained_adaptive_tcp/stream_demo.json")
    checkpoint = read("results/pretrained_native/checkpoint_verification.json")
    resume = read("results/pretrained_native/resume_replay_verification.json")
    quality = read("results/pretrained_native/quality_audit_final.json")
    coverage = read("results/pretrained_native/coverage_final.json")["totals"]["percent_covered"]
    mutations = read("results/mutations/summary.json")
    updates = training_updates()
    test_log = (OUTPUT / "pytest_final.txt").read_text()
    tests = int(re.findall(r"(\d+) passed", test_log)[-1])
    rows, table = budget_table(initial, trained)
    figures(initial, trained, native, updates)
    native_psnr = mean(values(native["records"], "psnr"))
    native_bpp = mean(values(native["records"], "total_bytes")) * 8 / (1920 * 1080)
    cache_error = maximum_cache_error(long["records"])
    summary = dict(
        tests_passed=tests,
        coverage_percent=coverage,
        mutations_killed=sum(row["killed"] for row in mutations),
        mutations_total=len(mutations),
        trained_steps=100,
        native_teacher_frames=checkpoint["native_teacher_frames"],
        official_backbones_unchanged=checkpoint["official_backbones"],
        checkpoint=checkpoint,
        resume_replay=resume,
        native_baseline_psnr=native_psnr,
        native_baseline_bpp=native_bpp,
        codebook_collapsed=all(frame["vq_perplexity"] == 1.0 for frame in updates[-1]["records"]),
        trained_rd_validated=False,
        long_sequence_frames=long["sample_count"],
        maximum_cache_pixel_error=cache_error,
        budget_summary=rows,
        two_prefix_fps=two["overall"]["fps"],
        five_prefix_fps=five["overall"]["fps"],
        native_codec_fps=native_speed["overall"]["fps"],
        target_30fps_met=two["target_30fps_met"],
        target_60fps_met=two["target_60fps_met"],
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = f"""# 官方骨干与原生码流后续报告

官方权重、原生 CUDA 码流和真实 decoded-latent 蒸馏已跑通，完成100步 Stage1及实际1080p评估。**当前码本塌缩，压缩质量不合格，30/60 FPS目标仍未达成。** 本报告记录短程工程实验，不代表研究目标或全数据集训练已经完成。历史随机骨干结果保留在 [原报告](implementation_report.md)。

## 环境与 teacher 修复

独立 `.venv-native`：Python3.12.13、PyTorch2.9.1+cu130、nvcc13.0.88、CUTLASS4.4.1、H100 PCIe。环境版本、补丁及编译源 SHA256 见 `results/native_environment.json`，依赖见 `native_environment_freeze.txt`。原有 Python3.10/torch2.6 环境及 InfoTok 项目没有修改。

原先未执行过的 PyTorch prior 重算 teacher 在真实 native 核对中失败：最大像素误差0.05914，大于0.02门限。已删除这条近似路径，保留失败日志 `native_gate_red.txt`。现在独立 native encoder 生成 SPS/IP 码流，独立 native decoder 消费这些字节，构建副本上的 [最小补丁](../native_patches/decoded_latent.patch) 返回解码器实际使用的 `m_y_hat` 的 detached clone。没有改变原生熵编码、量化或参考算法；Git submodule 源码保持原样。I/P目标精确等于原生 latent；修改返回副本不会改变内部状态；重置后 I/P 字节完全一致。

## 训练与 checkpoint

- 官方 LD375个和 DMCI398个 state tensor 均与下载文件逐项完全一致；仅训练新增 queries、VQ、统一 decoder。
- N_max2048、dim192、6heads、2layers、VQ8192×16、crop256、每次4帧、bf16、TBPTT2、LPIPS和真实 native latent loss；100次更新共{checkpoint["native_teacher_frames"]}个 teacher 帧。
- MCL-JCV01训练64帧、UVG Beauty验证16帧、BasketballDrive测试32帧。无 resize；全部112帧的 RGB 与原始 YUV 经显式 BT.709 limited-range 转换所得字节哈希一致。原始 YUV 无颜色元数据，该转换规则作为假设记录。
- 使用checkpointed generator采样连续帧窗口，100步访问{len(checkpoint["sampled_starts"])}个起点；记录每帧源位置。全1080p验证已执行。
- 保存 `checkpoints/pretrained_s1/step100.pt`，SHA256 `{checkpoint["sha256"]}`。`latest.pt` 已恢复到101步，100步文件不变。
- 两个独立进程从同一100步checkpoint恢复到101步后，采样/CPU/CUDA/Python RNG全部一致。CUDA bf16的模型和optimizer未达到逐位相同，模型最大绝对差 `{resume["maximum_model_abs_error"]:.9g}`；CPU恢复行为测试通过。不能声称 GPU 训练逐位可复现。

## 实际质量：同一独立8帧，五档真实字节重建

{chr(10).join(table)}

最大档初始 LPIPS {initial["budget_summary"][-1]["lpips"]:.4f}→{trained["budget_summary"][-1]["lpips"]:.4f}，PSNR {initial["budget_summary"][-1]["psnr"]:.4f}→{trained["budget_summary"][-1]["psnr"]:.4f}dB。预算增加而 MSE 变差的相邻对比例为{trained["quality_drop_ratio"]:.2%}；微小差异不能当作有效渐进质量改善。

第100步四帧的 VQ perplexity均为1、使用率1/8192。当前新增瓶颈已塌缩，所有前缀的重建近乎相同；延长训练前应优先解决该训练问题，并重新检查实际 RD。Scorer和熵模型尚未做这一预训练骨干版本的 Stage3校准。

官方原生 QP32同8帧实际 SPS/IP码流 PSNR **{native_psnr:.4f}dB**、bpp **{native_bpp:.8f}**。这是单个不同码率工作点，不能计算 BD-rate或声称等率比较。CP列统计真实 length-framed records，native列统计 SPS/IP bytes，未混入假设的物理链路开销。

32帧检查的全部五档经过真实字节解码，缓存/完整重建最大像素差 **{cache_error:g}**，独立基础参考状态检查通过。该检查证明时序与缓存正确，不证明画质。

## 性能与 TCP

CP基于同一100步checkpoint，1920×1080、8测量帧、2 warmup，无coverage插桩、串行测量：

| 路径 | 整体 FPS | 稳态P FPS | P95 ms |
|---|---:|---:|---:|
| CP基础+最大 | {two["overall"]["fps"]:.2f} | {two["steady_inter"]["fps"]:.2f} | {two["overall"]["p95_ms"]:.2f} |
| CP五档显示 | {five["overall"]["fps"]:.2f} | {five["steady_inter"]["fps"]:.2f} | {five["overall"]["p95_ms"]:.2f} |

CP计入完整序列编码、真实熵码流、发送端本地基础解码/提交和接收端所有显示点重建。没有达到1080p30/60FPS。

原生独立encoder/decoder控制，8测量帧、4 warmup，整体{native_speed["overall"]["fps"]:.2f}FPS、稳态P {native_speed["steady_inter"]["fps"]:.2f}FPS；计入RGB转换、SPS/IP封装解析和原生编解码，排除初始化、源I/O和指标。原生使用官方FP16 fused CUDA算法与一次重建，CP使用PyTorch和多显示点，二者工作量不同；原生速度不代表CP达标。

另有2帧五档PNG真实TCP，以及8帧 Rate-only、当前32/128/512/2048/128/512/2048/512kbps、1200ms源截止时间、增强最多512的真实TCP实验。后者两端各提交{tcp["sender_base_commits"]}个相同基础参考，省去{tcp["omitted_enhancement_packets"]}个增强包，{tcp["base_wait_frames"]}个基础ACK超时，真实显示迟到{tcp["deadline_missed_displays"]}次。准确记录冷启动和积压，不保证实时交付；应用层整形不代表物理丢包实验。

## 测试、审计与复现

- native环境完整 pytest：{tests} passed；旧环境结果见 `results/pretrained_native/pytest_legacy.txt`。
- 当前源码独立覆盖率数据库：{coverage:.2f}%；最大生产函数复杂度{quality["max_complexity"]}；CRAP≥10条目{len(quality["crap_over_10"])}；AST重复≥0.8条目{len(quality["duplicates_over_08"])}。未合并改行前数据库。命名审计工具不可用，仍使用明确记录的替代工具。
- 隔离行为变异{sum(row["killed"] for row in mutations)}/{len(mutations)} killed。真实CLI及性能命令、退出码分别见 `final_commands.json`、`measurements_commands.json`；覆盖率插桩结果不用于性能声明。

```bash
cd /home/whisperliang/CP-AdapTok
bash scripts/setup_native.sh
source .venv-native/bin/activate
python scripts/prepare_pretrained_data.py
python -m pytest cp_adaptok/tests -q
# 首次新训练；已完成的路径应使用下面的resume命令继续
python -m cp_adaptok.cli.train --config configs/pretrained_s1.yaml --max-steps 100
python -m cp_adaptok.cli.train --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --resume --max-steps 101
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --max-frames 8 --output results/pretrained_s1_test
python -m cp_adaptok.cli.eval --config configs/pretrained_s1.yaml --baseline native --max-frames 8 --output results/pretrained_native_baseline
python -m cp_adaptok.cli.benchmark --config configs/pretrained_s1.yaml --checkpoint checkpoints/pretrained_s1/step100.pt --resolution 1920 1080 --measure actual-bitstream --frames 8 --warmup 2 --output results/pretrained_benchmark_two
python -m scripts.benchmark_native
python scripts/build_pretrained_report.py
```

首次训练会读取已有 `checkpoint` 路径作为warm start；复现实验时应保留并使用选定的100步文件，或另设空的checkpoint/输出路径。当前训练量不足，码本塌缩、后续阶段、完整RD与空间/AdapTok对照、Scorer泛化、CP实时优化仍待完成。详见 [ub-mode操作日志](work_log.md)。

![短程训练与码本塌缩]({OUTPUT / "figures/pretrained_quality.png"})
"""
    (ROOT / "docs/pretrained_native_report.md").write_text(text)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("checkpoint", "budget_summary", "official_backbones_unchanged")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
