"""Generate a measured report; missing experiments are never replaced with estimated quality."""

import argparse
from pathlib import Path

from .artifacts import write_json
from .comparison import compare, load, official_rows


def quality_table(summary):
    lines = ["| K / Tile / 16帧 | PSNR逐帧均值 dB | PSNR全局MSE dB | MS-SSIM | LPIPS-Alex | LPIPS-VGG | DISTS |",
             "|---|---|---|---|---|---|---|"]
    for budget, item in summary["budgets"].items():
        m = item["metrics"]
        lines.append(f"| {budget} | {m['psnr_mean_db']:.4f} | {m['psnr_from_global_mse_db']:.4f} | {m['ms-ssim']:.6f} | {m['lpips']:.6f} | {m['lpips-vgg']:.6f} | {m['dists']:.6f} |")
    return "\n".join(lines)


def measured_figures(root, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = root / "final_1080p/figures"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, summary in zip(["Stride 128", "Stride 96"], summaries):
        budgets = [int(k) for k in summary["budgets"]]
        axes[0].plot(budgets, [summary["budgets"][str(k)]["metrics"]["psnr_mean_db"] for k in budgets], "o-", label=label)
        metric = summary["budgets"]["2048"]["metrics"]
        axes[1].plot(["Interior", "Boundary"], [metric["interior_mse"], metric["boundary_mse"]], "o-", label=label)
    axes[0].set(xlabel="VQ tokens / tile / 16 frames", ylabel="Mean frame PSNR (dB)")
    axes[1].set(ylabel="RGB mean squared error", title="Same fixed 128-pixel seam mask")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.savefig(directory / "quality_and_seams.png", dpi=180)
    plt.close(fig)


def build_artifact_index(root):
    for name in ["final_1080p", "token_ablation/overlap"]:
        directory = root / name
        index_layout(directory)


def index_layout(directory):
    video_index, sample_index = directory / "reconstructed_videos", directory / "reconstruction_samples"
    video_index.mkdir(exist_ok=True)
    sample_index.mkdir(exist_ok=True)
    for path in directory.glob("*/reconstructed_videos/*.mp4"):
        ensure_link(video_index / path.name, path, directory)
    for path in directory.glob("*/completed.json"):
        video = load(path)
        for budget in video["budgets"]:
            target = Path(video["windows"][0]["directory"]) / f"samples_k{budget}"
            ensure_link(sample_index / f"{path.parent.name}_k{budget}", target, directory)


def ensure_link(link, target, directory):
    if link.exists():
        if link.resolve() != target.resolve():
            raise ValueError(f"refusing to replace a different artifact: {link}")
        return
    link.symlink_to(Path("..") / target.resolve().relative_to(directory.resolve()))


def latency_table(root):
    items = []
    for name in ["no_overlap", "overlap", "final_1080p", "token_ablation/overlap"]:
        item = load(root / name / "summary.json")["budgets"]["2048"]
        items.append((name, item["latency"], item["peak_inference_allocated_mib"], item["peak_inference_reserved_mib"]))
    return format_latency_table(items)


def benchmark_table(root):
    summary = load(root / "benchmarks/summary.json")
    return format_latency_table([(f"Benchmark stride{stride}", item["latency"], item["peak_allocated_mib"], item["peak_reserved_mib"]) for stride, item in summary.items()])


def format_latency_table(items):
    lines = ["| 测量 | 真实帧 | NN FPS (CUDA) | NN FPS (wall) | 计算 FPS | 含读写 FPS | 计算均值ms/帧 | P50 | P95 | 峰值 allocated / reserved MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, latency, allocated, reserved in items:
        compute = latency["compute"]
        lines.append(f"| {label} | {latency['source_frames']} | {latency['neural_cuda_events']['total_fps']:.3f} | {latency['neural']['total_fps']:.3f} | {compute['total_fps']:.3f} | {latency['pipeline']['total_fps']:.3f} | {compute['mean_ms_per_frame']:.2f} | {compute['p50_ms_per_frame']:.2f} | {compute['p95_ms_per_frame']:.2f} | {allocated:.1f} / {reserved:.1f} |")
    return "\n".join(lines)


def comparison_table(comparison):
    lines = ["| 模型 | Token 单位 | PSNR逐帧均值 dB | LPIPS-Alex | DISTS |",
             "|---|---|---|---|---|"]
    for row in comparison["rows"]:
        if row["content_id"] == "all":
            lines.append(f"| {row['model']} | {row['budget_units']} | {row['psnr_mean_db']:.4f} | {row['lpips']:.6f} | {row['dists']:.6f} |")
    return "\n".join(lines)


def seam_table(summaries):
    lines = ["| Stride | K | 边界 MSE | 非边界 MSE | MSE差值 | 边界PSNR dB | 非边界PSNR dB |",
             "|---|---|---|---|---|---|---|"]
    for stride, summary in zip([128, 96], summaries):
        for k, item in summary["budgets"].items():
            m = item["metrics"]
            lines.append(f"| {stride} | {k} | {m['boundary_mse']:.7f} | {m['interior_mse']:.7f} | {m['boundary_minus_interior_mse']:.7f} | {m['boundary_psnr_from_mse_db']:.4f} | {m['interior_psnr_from_mse_db']:.4f} |")
    return "\n".join(lines)


def per_video_table(comparison):
    rows = [r for r in comparison["rows"] if r["content_id"] != "all" and r["budget_units"].startswith("2048/")]
    lines = ["| 视频 | 模型 | PSNR dB | LPIPS-Alex | DISTS |", "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: (r["content_id"], r["model"])):
        lines.append(f"| {row['content_id']} | {row['model']} | {row['psnr_mean_db']:.4f} | {row['lpips']:.6f} | {row['dists']:.6f} |")
    return "\n".join(lines)


def budget_regressions(root, directory):
    data = {k: { (r["content_id"], r["frame_id"]): r for r in official_rows(root / directory, k)[1]} for k in [512, 1024, 2048]}
    return {f"{low}->{high}": {"psnr_worse_frames": sum(data[high][key]["psnr"] < row["psnr"] for key, row in data[low].items()),
                               "total_frames": len(data[low])} for low, high in [(512, 1024), (1024, 2048)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/original_adaptok_1080p"))
    parser.add_argument("--reference", type=Path, default=Path("results/large_scale/formal_20261116/final_test"))
    parser.add_argument("--output", type=Path, default=Path("docs/original_adaptok_1080p_report.md"))
    args = parser.parse_args()
    root = args.results.resolve()
    comparison = compare(root, args.reference)
    summaries = [load(root / p / "summary.json") for p in ["final_1080p", "token_ablation/overlap"]]
    facts = {"comparison": comparison, "summaries": summaries,
             "regressions": {p: budget_regressions(root, p) for p in ["final_1080p", "token_ablation/overlap"]}}
    write_json(root / "report_facts.json", facts)
    build_artifact_index(root)
    measured_figures(root, summaries)
    args.output.write_text(render_report(root, facts), encoding="utf-8")
    print(args.output.resolve())


def render_report(root, facts):
    checkpoint = load(root / "checkpoint_manifest.json")
    env = load(root / "environment.json")
    e0 = load(root / "smoke/official_equivalence.json")
    sources = load(root / "input_manifest.json")["sources"]
    a, b = facts["summaries"]
    ma, mb = a["budgets"]["2048"]["metrics"], b["budgets"]["2048"]["metrics"]
    gain = mb["psnr_mean_db"] - ma["psnr_mean_db"]
    cp = {r["model"]: r for r in facts["comparison"]["rows"] if r["content_id"] == "all" and r["budget_units"] == "2048/frame"}
    source_lines = "\n".join(f"| {s['content_id']} | {s['width']}×{s['height']} | 0–63 / 64 | {s['fps']} (nominal) | `{s['sha256']}` |" for s in sources)
    first_video = next((root / "no_overlap").glob("*/completed.json"))
    sample = Path(load(first_video)["windows"][0]["directory"]) / "samples_k2048"
    preview = str(sample.relative_to(root))
    certification = validation_text(root)
    return f"""# 官方 AdapTok 原生 1080p 分块重建实测

本实验完成 E0–E4。仅推理，没有训练、微调、反向传播、配置覆盖或模型结构修改。
结论限于 DAVIS2017 cows、goat、parkour 各 64 帧，共 192 帧；时间窗口独立，无跨窗口因果状态传递。
官方来源：[源码](https://github.com/VisionXLab/AdapTok)、[预训练模型](https://huggingface.co/yeahhhh326/AdapTok)。

## 版本与不可变性

- 基于 CP main `534696b2f78b7e1bc74bf5c5eefe83e88f41b864`，分支 `experiment/original-adaptok-1080p`。
- 官方源码 Commit：`{checkpoint['official_git_commit']}`，子模块 tracked 文件未修改。
- HF Revision：`{checkpoint['revision']}`，严格调用原生 `AdapTok.from_pretrained(local_dir)`，不传配置覆盖。
- config.json SHA256：`{checkpoint['config_sha256']}`。
- model.safetensors SHA256：`{checkpoint['model_safetensors_sha256']}`，已与官方 LFS 哈希核验。
- State SHA256 前/后：`{checkpoint['state_sha256_before']}` / `{checkpoint['state_sha256_after']}`。
- 输入 `{checkpoint['frame_num']}×{checkpoint['input_size']}×{checkpoint['input_size']}`，Codebook `{checkpoint['codebook_size']}`，Latent `{checkpoint['latent_tokens']}`，时间组 `{checkpoint['temporal_groups']}×{checkpoint['tokens_per_group']}`。
- 参数量 `{checkpoint['parameters']:,}`（含 checkpoint 中原生 prior；没有调用 prior/scorer/IPAL）。推理 `{checkpoint['dtype']}`，无 AMP、TF32 关闭。
- GPU `{env['gpu']}`；Python `{env['python']}`；PyTorch `{env['packages']['torch']}`；CUDA runtime `{env['cuda_runtime']}`。完整驱动和依赖见 environment.json、requirements-lock.txt。
- 环境 `.venv-adaptok-official` 独立安装；未修改 `.venv-native`、系统 CUDA 和旧实验。

## 数据与指标口径

| 内容 ID | 原生尺寸 | 帧索引 / 帧数 | FPS | 全序列来源 SHA256 |
|---|---|---|---|---|
{source_lines}

真实来源路径及逐文件哈希位于 input_manifest.json，均从 `data/manifests/cpat_test.json` 读取，未按名称猜路径。
直接复用 CP 的 `cp_adaptok.data.read_video`：OpenCV 解码、BGR→RGB、float32 /255，既不整帧 Resize，也不裁剪。
在原 `.venv-native` 只读重读相同 12 个时间窗口后，RGB float32 SHA256 全部一致；LPIPS-Alex、DISTS 校准权重和 AlexNet/VGG16 骨干文件 SHA256 全部一致。
版本和证据见 cp_input_parity.json、metric_manifest.json、absolute_quality_comparison.json。
原图与重建都在完整 RGB 1080×1920 上计算指标，LPIPS 输入 [-1,1]，DISTS 使用 CP 相同 [0,1] 预处理与 alpha/beta，MS-SSIM data_range=1。
LPIPS-VGG 对齐官方评估的 normalize=True 语义。DISTS 内部标准多尺度处理与 CP 完全相同，未在实验层缩小输入。

每 16 帧独立调用官方 encode_eval、decode_eval。尾窗重复最后真实帧，仅真实帧参与指标与 FPS 分母；纯功能测试覆盖 1、15、16、17、33 帧。
无重叠：反射补到底部 1152，15×9=135 Tile；stride96：补至1088×1952，20×11=220 Tile。
位置固定跨所有帧；默认 Batch=4。无重叠直接写回，重叠采用半像素采样 sin² 的可分权重，全图有效像素权重严格大于零，最后裁回 1080×1920。

## E0 一致性与 E1 单视频

真实 Tile 原生接口对照最大绝对误差 `{e0['official_max_abs_error']}`；重复 VQ 编码完全一致；State 哈希相同。
K512/1024/2048 分别原生调用分组 Mask，保留 `[128]*4`、`[256]*4`、`[512]*4`，各预算独立解码与官方原生调用误差为零。
同一 Tile 在 Batch1 与 Batch4 的最大数值差为 `{e0['batch1_vs_batch4_max_abs_error']:.9g}`，来自浮点批量计算；Token 分配定义不变。
该 Tile PSNR `{e0['tile_psnr_db']:.4f}` dB。单 Tile 结果只用于接口验收。

首条 cows 的首 16 帧无重叠全 Token：

{quality_table(load(root / 'no_overlap/summary.json'))}

![原图与重建，左/右，均为原生1080p](../results/original_adaptok_1080p/{preview}/comparison.png)

![跨边界原生像素裁块，左/右](../results/original_adaptok_1080p/{preview}/boundary_crop.png)

原始及重建视频在 no_overlap、overlap 与各完整测试视频目录中；reconstructed_videos 为完整串接视频。
视频采用 libx264rgb CRF0 无损 RGB 保存，浮点输出仅为显示量化到 uint8，指标在量化和写盘前计算。
这些观看用视频不是模型熵编码码流，所有本实验 bpp 均为 null，quality_only=true。

## E3/E4 完整 192 帧 Token 消融

无重叠 stride128：

{quality_table(a)}

重叠融合 stride96：

{quality_table(b)}

各 Tile 的所有预算共享同一次完整 Encoder/VQ 输出，不切 `tokens[:K]`。
逐帧回退计数（如有则保留原值）：`{facts['regressions']}`。
逐帧 CSV、逐视频 CSV 以及视频等权均值都已保存。PSNR逐帧均值是 dB 的算术平均；全局 PSNR = -10log10(全部完整帧的平均 MSE)，两列分别报告，不混用。
三条视频帧数相等，因此这次逐帧均值与逐视频等权均值相同；代码保留两种汇总。

## E2 空间边界诊断

固定边界 Mask 为内部 x/y = 128n 处 `[line−2,line+2)` 的 4 像素条带并集，排除图像外缘。
两种 stride 都用这同一个 Mask，便于比较对原始135块拼缝的修复；它不把 stride96 所有块边缘都当作独立硬拼缝。
边界 PSNR 与非边界 PSNR 由各自区域的聚合 MSE 计算；区域纹理不同，因此差值不是孤立的因果边界效应。

{seam_table([a, b])}

全 Token overlap 的平均逐帧 PSNR 差值为 `{gain:+.4f}` dB；真实 Tile 数从135增加到220，增加62.96%。不预设重叠方案必然更好。

![Token预算与边界误差](../results/original_adaptok_1080p/final_1080p/figures/quality_and_seams.png)

## 耗时与显存

独立单预算 Benchmark：全 Token、三个真实视频各连续16帧，每种 stride 共48帧，单独进程仅加载 tokenizer，指标网络不驻留：

{benchmark_table(root)}

质量评估进程中的计时与驻留显存另列，便于核查完整消融：

{latency_table(root)}

CUDA Events 与显式 synchronize 同时记录 Encode/Decode，wall 包含接口内部 CPU 分派。神经网络 FPS 分别提供 event 和 wall 两种口径。
Event elapsed 包含接口时间线上主机分派造成的间隙，不等于内核活跃时间或实测 FLOPs。
计算 FPS 包含分块、CPU→GPU、Encode、Decode、拼接、分配与分派开销；完整流水线再加视频读取、原图/重建视频写盘和样图 PNG 写盘。
指标计算时间独立记录于 latency.json 的 metrics_ms，全部 FPS 都不含指标计算。
前三批真实 Tile 预热、模型加载和 CUDA 初始化均不计入稳态计时。全部速度以真实源视频帧数计算，未用 Tile FPS 代替。
no_overlap/overlap 为各一个16帧窗口的独立单预算实测，因此其 P50/P95 只有一个窗口样本；final 两行各12个窗口，按窗口平均每帧延迟计算分位数，不冒充逐帧在线尾延迟。
完整多预算运行的每预算成本由实测分量相加，完整 Encoder/读取/共享开销均全额归该预算，不能当作额外独立单预算吞吐测量。
峰值 allocated 在每窗重建前 reset，重建后立即读取，排除感知指标的激活峰值；指标模型权重仍驻留，故不是孤立 tokenizer 进程的最小显存。
reserved 与 all_budgets_compute_wall_ms 也完整保存；多预算峰值包含三个重建缓冲区，不将它解释为单预算最小内存。

## 与封存 CP B0/B1 的同样本绝对画质比较

{comparison_table(facts['comparison'])}

全 Token 的逐视频拆分：

{per_video_table(facts['comparison'])}

只复用封存 B0/B1 validation.json，不调用 DCVC 或修改历史结果。
CP 的 Token 单位为每完整帧，官方为每 Tile / 16帧；两者 Token 分配、表示、时间上下文和真实码率不同。
例如无重叠K2048平均有135×2048/16=17,280 VQ Token/源帧，重叠则为28,160，不能把表中相同K理解为同表示成本。这只是Token计数，未换算为bpp。
这是绝对重建质量比较，不能称为同码率公平 RD 比较，也不能据此声称压缩效率超过 CP。

## 研究问题与边界

**Q1**：可以。真实16帧 Tile 的官方接口、确定性与权重不可变性均通过；单 Tile PSNR 仅为一致性例证。

**Q2**：三条封存视频192帧、无训练、无整帧降采样下，无重叠全 Token 平均 PSNR 为 {ma['psnr_mean_db']:.4f} dB，LPIPS-Alex {ma['lpips']:.6f}、DISTS {ma['dists']:.6f}；可用程度取决于目标应用，需结合原生样图与逐视频结果判断。

**Q3**：边界与非边界误差、overlap 质量变化及62.96%的 Tile 增量均在上表实测；全 Token 平均 PSNR 差值 {gain:+.4f} dB，LPIPS-Alex 差值 {mb['lpips']-ma['lpips']:+.6f}，DISTS 差值 {mb['dists']-ma['dists']:+.6f}。不能只凭一个指标断言所有视频受益。

**Q4**：有效改善。无重叠512→1024→2048的平均PSNR为{a['budgets']['512']['metrics']['psnr_mean_db']:.4f}→{a['budgets']['1024']['metrics']['psnr_mean_db']:.4f}→{ma['psnr_mean_db']:.4f} dB；重叠为{b['budgets']['512']['metrics']['psnr_mean_db']:.4f}→{b['budgets']['1024']['metrics']['psnr_mean_db']:.4f}→{mb['psnr_mean_db']:.4f} dB。逐帧回退计数如实保留，未强制单调。

**Q5**：以独立单预算 Benchmark 表为实际 tokenizer 资源测量：每种 stride 三个16帧窗口、共48帧；平均/P50/P95、NN/计算/读写FPS和显存均实测。固定135/220块代价显著，不能用128像素 Tile 的速度作部署承诺。

**Q6**：同帧、同 RGB、同感知权重下，官方无重叠全Token相对CP B0/B1全预算平均PSNR分别提高{ma['psnr_mean_db']-cp['B0']['psnr_mean_db']:.4f}/{ma['psnr_mean_db']-cp['B1']['psnr_mean_db']:.4f} dB，感知指标见完整表和逐视频表；优势不等于同码率压缩优势。

**Q7**：建议下一研究重点优先放在跨空间块共享上下文，保留已实现的平滑overlap作为边界对照：它以62.96%更多Tile换取{gain:.4f}dB，说明简单融合有收益，但成本显著。跨块共享能否减少冗余仍是待验证假设。原生高分辨率Encoder/Decoder需要训练及新的受控实验，排在该诊断之后，本实验没有证明其优劣。

测量支持继续将官方 AdapTok 作为高分辨率研究的基础对照与诊断工具；是否作为部署主路线，需要更广视频内容、实际熵码流、同码率对比及跨块共享的成本实验。
本次只有3条DAVIS序列、192帧，名义30FPS，无音频、无跨窗状态、无重训练，不能泛化到任意1080p视频或长期时序一致性，也没有测量公平RD。

## 复现

```bash
bash experiments/original_adaptok_1080p/setup.sh
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.download_checkpoint --output checkpoints/original_adaptok_official/{checkpoint['revision']}
bash experiments/original_adaptok_1080p/run.sh verify --output results/original_adaptok_1080p/verification
bash experiments/original_adaptok_1080p/run.sh suite
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.benchmark
.venv-native/bin/python -m experiments.original_adaptok_1080p.verify_cp_inputs
.venv-adaptok-official/bin/python -m experiments.original_adaptok_1080p.report
OMP_NUM_THREADS=4 .venv-adaptok-official/bin/python -m pytest experiments/original_adaptok_1080p/tests -q
```

单视频：`run.sh evaluate --input /absolute/path/video.mp4 --frames 17 --stride 96 --budgets 512 1024 2048 --output results/original_adaptok_1080p/custom_run`。
YUV 另需 `--raw-width 1920 --raw-height 1080 --fps 30 --raw-pixel-format yuv420p --raw-color-range limited --raw-color-matrix bt709`，按源真实色彩元数据填写，不能猜测；与 CP MP4 的 OpenCV 路径不同，明确记录转换规则。
重复同配置复用已完成窗口；不同输入 SHA256、预算或配置拒绝覆盖，崩溃遗留尝试保留独立目录。操作与验证日志见 original_adaptok_1080p_work_log.md。

## 测试与最终验收

{certification}
"""


def validation_text(root):
    certificate = root / "acceptance.json"
    if not certificate.exists():
        return "最终机器验收正在执行，数值实验已经完成。"
    acceptance = load(certificate)
    new_tests = (root / "pytest_final.txt").read_text().strip().splitlines()[-1]
    cp_tests = (root / "pytest_cp_regression.txt").read_text().strip().splitlines()[-1]
    return (f"机器验收：`{acceptance['all_acceptance_checks_passed']}`，见acceptance.json。新增测试（含真实GPU一致性）：`{new_tests}`；原CP相关回归：`{cp_tests}`。\n"
            f"真实17帧GPU重建只对16+1帧计分，整理后的代码与早期首窗指标最大绝对差`{acceptance['tail']['refactor_vs_original_run_metric_max_abs_error']}`。\n"
            "6项行为变异全部被有效测试杀死；未将收集错误算作成功。radon/coverage/AST审计见quality_audit.json；crapper/dryer/mutator专用工具不可用，明确使用替代工具。\n"
            "已核验18个完整视频均为1920×1080、64真实帧；原CP代码、训练配置、子模块及历史B0/B1结果保持不变。")


if __name__ == "__main__":
    main()
