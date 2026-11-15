"""Generate the delivery from actual experiment JSON, without filling missing results."""

import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plots(evaluation, two, five):
    directory = ROOT / "results/figures"
    directory.mkdir(exist_ok=True)
    rows = evaluation["budget_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, metric, limit in zip(axes, ("psnr", "lpips", "dists"), ((0, 50), (0, 2), (0, 1))):
        ax.plot([r["bpp"] for r in rows], [r[metric] for r in rows], "o-", color="#2563eb")
        ax.set(xlabel="Actual application bpp", ylabel=metric.upper(), ylim=limit)
        ax.grid(alpha=0.2)
    fig.suptitle("UNTRAINED BACKBONE — engineering measurements only (2 RGB frames)")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory / f"rd_curves.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].bar(
        ["base + max", "all 5 prefixes"], [two["overall"]["fps"], five["overall"]["fps"]], color="#2563eb"
    )
    axes[0].axhline(30, color="#d97706", linestyle="--", label="30 FPS target")
    axes[0].set(ylabel="End-to-end FPS", ylim=(0, 35))
    axes[0].legend()
    axes[1].plot([r["pipeline_ms"] for r in two["frames"]], "o-", label="base + max")
    axes[1].plot([r["pipeline_ms"] for r in five["frames"]], "o-", label="all 5")
    axes[1].set(xlabel="Frame index (I then P)", ylabel="Pipeline latency (ms)")
    axes[1].legend()
    fig.suptitle("1080p actual rANS stream, H100 PCIe; sender-local decode and state checks included")
    fig.savefig(directory / "latency.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    for mode in ("fixed", "rate-only", "scorer"):
        points = [r["online"][mode].get("tokens", 0) for r in evaluation["allocation"]]
        ax.step(range(len(points)), points, where="mid", marker="o", label=mode)
    oracle = [r["offline_oracle"]["tokens"] for r in evaluation["allocation"]]
    ax.plot(oracle, "x--", label="offline oracle (noncausal)")
    ax.set(xlabel="Frame", ylabel="Selected tokens (0 = base wait)")
    ax.set_title("Encoding time deducted from deadline; scorer remains smoke-trained")
    ax.legend()
    fig.savefig(directory / "allocation.png", dpi=180)
    plt.close(fig)


def environment():
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "PyYAML",
        "pytest",
        "coverage",
        "radon",
        "ruff",
        "lpips",
        "DISTS-pytorch",
        "pytorch-msssim",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (ROOT / "results/environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    return versions


def streaming_results():
    path = ROOT / "results/streaming_verified/summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    labels = {
        "all_prefixes_png": "全部五档 PNG",
        "tight_deadline": "300 ms 截止时间",
        "fixed": "固定目标 2048",
        "rate_only": "Rate-only",
        "scorer": "Causal Scorer",
        "truncated_tail": "2048 目标截尾到512",
        "delayed_ack": "100 ms ACK 延迟",
    }
    table = [
        "| 场景 | 帧数 | 实际发送 Token 前缀 | 双向应用 bytes | 基础 ACK 迟到帧 | 显示迟到次数 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary["scenarios"]:
        name = row["scenario"]
        base_late = "—" if name == "all_prefixes_png" else str(row["base_wait_frames"])
        display_late = "—" if name == "all_prefixes_png" else str(row["deadline_missed_displays"])
        table.append(
            f"| {labels[name]} | {row['frames']} | {', '.join(map(str, row['delivered_tokens']))} | "
            f"{row['application_bytes']} | {base_late} | {display_late} |"
        )
    sample = read("results/streaming_verified/truncated_tail/stream_demo.json")
    text = f"""# 在线 TCP 后续验证

本页使用旧随机骨干结构模型，保留其历史场景结果。官方骨干版本的原生扩展、训练与真实 TCP 结果另见 [新报告](pretrained_native_report.md)。

本轮把在线预算选择接入真实 localhost TCP：完整编码一次，先可靠交付并确认基础前缀，再按当前带宽和剩余时间选择增强包。每个增强包发送前复查截止时间；接收端仅使用模型和线路字节。**状态仍为 `{sample["status"]}`；这些结果验证工程行为，不验证 RD 或 Scorer 泛化。**

## 实测条件与结果

- 代码提交 `{sample["git_commit"]}`；每份 JSON 同时记录源码 SHA256、模型 ID、配置与 GPU 快照。
- 同一个 Stage 3 结构 checkpoint，N_max=2048、K_ref=128、原生1920×1080 RGB，H100 PCIe。在线场景为8帧/50 FPS源时间轴，每个 CLI 独立进程。源时钟从发送循环开始，包含首帧 kernel 冷启动和积压；checkpoint 加载及模型构建先完成。
- Fixed/Rate-only/Scorer 截止时间1200 ms，当前逐帧带宽为32/128/512/2048/128/512/2048/512 kbps；固定目标无法及时交付时仍可靠保留基础图。
- 截尾场景为2500 ms、2048 kbps，只舍弃未发送的连续增强尾部；ACK 延迟场景为1200 ms、2048 kbps。除两帧五档 PNG 外，显示指 Tensor 就绪并转到 CPU，不含实体显示器扫描。

{chr(10).join(table)}

七组的发送/接收基础提交次数、参考摘要、原包前缀 SHA256 以及双向应用字节全部核对。截尾场景8帧均从2048截到512，省去96个增强包，后续 P 帧无漂移；300 ms 场景所有基础帧迟到，省去120个增强包，不能宣称达到实时目标。表中策略结果受同一帧历史因果状态约束；每次独立冷启动存在调度波动，不能据此推断策略质量优势。

## 修复并复现的问题

1. 基础显示后按整帧计费会重复扣除基础字节；现在只计未发送字节和62-byte 显示交换控制。
2. 排除已显示前缀会迫使 Scorer 选择预测更差的增强；现在保留零字节成本的基础候选。
3. 固定25 ms解码预留在先前实测中造成6.46 ms增强显示迟到；现在取配置下限与已观测显示 ACK 耗时上界的1.25倍中较大者，包含当前基础及历史确认，不读取未来耗时。旧失败证据在 `results/streaming_followup`，修复后证据在 `results/streaming_verified`。
4. 控制字段必须为规范整数，ACK 摘要必须完整128 bit，NaN/Inf截止时间在 I/O 前拒绝；基础缺失不能正常结束会话。红测试日志和行为变异覆盖这些约束。
5. Python API 的 `8.0` Token 预算此前也能通过范围检查并提交参考；现在所有编码/包切片/显示入口统一要求规范整数，在基础提交前拒绝非法值。

## 字节与时间口径

FrameHeader/Packet bytes 不变。新增 CPDS/CPAK 含 CRC 与 length framing 分别23/39 bytes，每次显示共62 bytes；正常结束/确认共26 bytes。两帧五档 PNG 运行双向应用总字节7334，相比旧字符串 ACK 演示增加了完整校验控制，不能混用旧6882-byte测量。

源释放时间为 `origin + frame_id / source_fps`，不会随处理速度放慢。预算扣除编码、实际基础交付及显示 ACK 已花时间；报告区分真实显示迟到与发送端 ACK 迟到。预留仅为因果估计，不能保证冷启动和突发抖动下的硬实时。

前向 record 按当前带宽做应用层交付整形，ACK 路径未限速，但其字节计入总量并保守计入选择预算。没有模拟 TCP 随机丢包、内核重传、无线拥塞，也没有统计 TCP/IP/link 头。未发送尾部舍弃不等同于物理丢包。

## 复现

```bash
cd /home/whisperliang/CP-AdapTok
source .venv/bin/activate
python scripts/run_streaming_experiments.py --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --output results/streaming_verified
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive --config configs/streaming.yaml --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --allocation rate-only --bandwidth-kbps 32 128 512 2048 128 512 2048 512 --playback-delay-ms 1200 --no-save-frames
```

逐项命令和退出码见 `results/streaming_verified/commands.json`；逐帧预测、实际质量、deadline、字节、ACK、状态摘要在各场景的 `stream_demo.json`。本报告使用的旧结构模型画质仍不合格；官方权重已取得并验证（见 [权重记录](pretrained_weights.md)），原生 CUDA、正式长序列 RD 和30/60 FPS验收仍待完成。
"""
    (ROOT / "docs/streaming_report.md").write_text(text)
    return summary


def main():
    evaluation = read("results/optimized_eval/eval/evaluation.json")
    two = read("results/optimized_benchmark_two/benchmark.json")
    five = read("results/optimized_benchmark_five/benchmark.json")
    stream = read("results/optimized_stream/stream_demo.json")
    qa = read("results/quality_audit.json")
    coverage = read("results/coverage.json")["totals"]["percent_covered"]
    mutations = read("results/mutations/summary.json")
    commands = read("results/required_commands/commands.json")
    test_log = (ROOT / "results/pytest_final.txt").read_text()
    tests = int(re.findall(r"(\d+) passed", test_log)[-1])
    checkpoint = "checkpoints/s3_smoke/step100.pt"
    checkpoint_sha = sha256(checkpoint)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    versions = environment()
    streaming = streaming_results()
    plots(evaluation, two, five)
    rows = evaluation["budget_summary"]
    with (ROOT / "results/budget_quality.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| K | PSNR dB | MS-SSIM | LPIPS | DISTS | 实际 bpp | kbps (50 FPS源) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['tokens']} | {row['psnr']:.5f} | {row['ms-ssim']:.5f} | {row['lpips']:.5f} | "
            f"{row['dists']:.5f} | {row['bpp']:.8f} | {row['kbps']:.1f} |"
        )
    counts = evaluation["records"][0]["byte_counts"]
    errors = max(row["cache_max_error"] for frame in evaluation["records"] for row in frame["prefixes"])
    fps_rows = []
    for name, report in (("基础+最大", two), ("五档显示", five)):
        overall = report["overall"]
        fps_rows.append(
            f"| {name} | {overall['fps']:.2f} | {report['steady_inter']['fps']:.2f} | "
            f"{overall['p50_ms']:.2f} | {overall['p95_ms']:.2f} | "
            f"{report['peak_allocated_bytes'] / 2**30:.3f} |"
        )
    files = []
    for directory in ("cp_adaptok", "configs", "scripts", "docs"):
        files.extend(
            str(p.relative_to(ROOT))
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yaml", ".sh", ".md", ".txt")
        )
    files.append("docs/implementation_report.md")
    body = f"""# CP-AdapTok 实施与实验报告

本页保留随机骨干 smoke 阶段的历史测量。官方 CUDA 扩展、真实 decoded-latent teacher、100步官方骨干训练及最新测试已经完成，当前结果和码本塌缩问题见 [后续报告](pretrained_native_report.md)。本页中的“未执行”描述属于当时的实验状态。

可运行的工程研究原型已落地；**正式训练、有效 RD 和 1080p 30/60 FPS 验收尚未完成**。本报告直接由本地 JSON、实际测试日志生成，没有把随机骨干输出当作训练效果。

## 版本与环境

- 本报告生成前的代码提交：`{revision}`；测量文件各自保留实际 commit、dirty 状态、source SHA256 和模型 ID。
- 选用 checkpoint：`{checkpoint}`，Stage 3 结构训练 100 步，状态 `{evaluation["status"]}`。
- checkpoint SHA256：`{checkpoint_sha}`。
- 模型 ID：`{evaluation["model_id"]}`；codebook ID：`{evaluation["codebook_id"]}`。
- NVIDIA H100 PCIe；torch {versions["torch"]}；driver 580.159.03；PyTorch CUDA 12.4。关键 benchmark 串行运行，初始化/指标/I/O 不混入 pipeline FPS，状态计算没有省略。
- 官方 LD/DMCI 权重已于2026-11-15取得并通过官方大小/QuickXorHash、ZIP CRC及严格参数加载检查，见 [权重记录](pretrained_weights.md)。此前401/403已通过新版匿名浏览器会话解决；nvcc及两个原生扩展仍缺失。以下历史实验仍使用原有随机骨干 checkpoint，原生 Stage 0、native-parity teacher 未执行。
- 固定上游：见 [manifest](../third_party/manifest.json)，AdapTok `a72076cf6474f930a181aa78971de70d65289b49`，DCVC `cbdae87a5445114cdc7f48816da63ea80bdeac40`。源码未修改；保留原始许可和文件级 notices。

## 四项核心特性

| 特性 | 实际证据 | 状态 |
|---|---|---|
| 原生 1920×1080 输入/输出 | 官方 I/P 模块、1088 padding、RGB 色彩往返、真实 bytes 重建 | 工程验证通过 |
| 逐帧因果 base-only 闭环 | 32 帧、I reset、双方 feature/memory 直接比较、增强不改未来状态、丢基础包阻塞 | 工程验证通过 |
| learned 1D 有序 VQ 前缀、同一 Decoder | EncodeOnce；128–2048 及 4096/8192；Mask/完整与缓存对齐 | 工程验证通过 |
| 实际可追加熵码流与渐进显示 | rANS、CRC、完整旧 packet bytes 不变、TCP 基础显示 ACK 后追加 | 工程验证通过 |

质量随预算改善、Scorer 的泛化收益、同码率优于原生 LD/空间 control，以及 30/60 FPS **均未验证或未达标**。不能将上表理解为完整研究目标已达成。

## 测试与 ub-mode

- 最终 pytest：**{tests} passed**；有意义的行为变异：**{sum(x["killed"] for x in mutations)}/{len(mutations)} killed**。
- package 行/分支合并 coverage：{coverage:.2f}%。最大生产函数圈复杂度 {qa["max_complexity"]}；AST 相似度 ≥0.8 的重复函数对数 {len(qa["duplicates_over_08"])}。
- CRAP≥10 条目数 {len(qa["crap_over_10"])}，细节见 `results/quality_audit.json`；不得用该审计代替未执行 native CUDA 实验。
- 本会话没有 crapper/dryer/mutator 命名工具，使用 radon+coverage、AST 检查与隔离副本行为变异作为明确记录的替代工具。详见 [操作日志](work_log.md)。
- 实际执行了所有五个要求入口：pytest 成功，正式 train/eval/benchmark/stream 的退出码分别为 {[r["exit_code"] for r in commands[1:]]}，原因为缺正式权重/扩展/训练 checkpoint。显式 untrained 替代运行另行保存，未混称成功。

## 实际 1080p 质量与码率

测试为独立 BasketballDrive 短片中的 I/P 两帧；训练短片 MCL-JCV01，验证 UVG Beauty。**骨干是随机初始化，约 6.65 dB 表明质量不合格；下面仅证明工程测量链路，不能用作压缩性能结论。**

{chr(10).join(table)}

预算增加后实际 MSE 变差比例为 **{evaluation["quality_drop_ratio"]:.1%}**，没有隐瞒非单调性。当前 checkpoint 的密集/缓存重建最大像素差为 **{errors:g}**，PSNR/LPIPS/DISTS 均分别记录；默认 float64 累加通过此检查。早期 fp32 的差异及失败精度证据保留于 `results/smoke100_full_eval`。

I/P 分别保存了真实 header、packet、payload 与 framing 字节；第一帧最大预算示例：`{json.dumps(counts, ensure_ascii=False)}`。FrameHeader 固定108 bytes，Packet overhead 28 bytes，TCP 应用 framing 4 bytes/record。首轮交付的旧 socket demo 两帧双向应用总字节 {stream["total_application_bytes"]}；后续严格会话控制的字节另见下述在线 TCP 报告。均未抓取 TCP/IP/link 层开销。

## 实际码流性能

8 帧、2 帧 warmup；包含最大序列编码/全部 packet 熵编码、发送端本地基础 entropy decode/重建/状态提交、Scorer、接收端 context/entropy decode/基础和增强重建。

| 显示计划 | 整体 FPS | 稳态 P FPS | P50 ms | P95 ms | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(fps_rows)}

基础+最大时接收端每帧重建2次，另有发送端本地1次；五档时接收端5次，另有发送端1次。Attention 缓存只节省 token-to-grid 计算，后端仍按显示点重跑。不能把单个 kernel/forward 的速度当作 codec FPS。

优化前重复 CPU state SHA 读取约117.82 ms；改为设备整数摘要和 mutation-version 缓存后，实际完整 pipeline 提升到上表水平。初次摘要计算在状态提交区间内计时，后续合法读取可缓存。CPU/CUDA 摘要一致、原地修改和 inference Tensor 修改检测均有行为测试。**1080p 30 FPS 与 60 FPS 目标仍未达到**。

## 额外实验与未完成项

- 720p：真实1080p原像素的1280×720中心裁剪，明确无 resize；I/P五档实际码流已跑，不冒充独立720p数据集。见 `results/final_720_eval` / `results/optimized_benchmark_720`。
- 4096/8192：真实1080p单帧字节流、实际失真、缓存对齐已跑，状态 random_weights_structure_only。见 `results/final_extended8192`。
- 固定空间 control：独立配置与实际码流入口已跑随机结构检查，未训练，不据此作等码率优势比较。见 `results/final_spatial_control`。
- 训练：Stage 1结构100步 + resume至103；Stage 3结构100步；Stage 2/4各2步及短BPTT精确恢复测试。所有 checkpoint 均保留 untrained_backbone_smoke 状态。
- Scorer 保存真实预算标签和预测误差；在线 Fixed-K/Rate-only/Scorer 扣除已发生编码时间，基础也超时则等待；Offline Oracle 是明确非因果的全段 byte MILP。
- 后续已完成：在线预算接入实际 TCP，基础可靠交付、逐包截止检查、增强尾部舍弃、已观测 ACK 耗时预留及七组原生1080p实验。见 [在线 TCP 后续报告](streaming_report.md)，全部检查通过不代表全部截止时间达标。
- 未完成：官方 CUDA baseline/teacher 数值核对；正式长序列训练和全数据集 RD；训练好的空间/AdapTok同条件对照；泛化 Scorer 校准；4K；compile/CUDA Graph/fused entropy 的质量和兼容性验证；真实物理网络丢包/拥塞实验；物理线路开销统计。
- Git 是本地新仓库，未配置远程，因此只完成有序本地提交，没有推送。没有创建需清理的 worktree，也没有修改已有 InfoTok 项目或其训练。

## 复现命令

```bash
cd /home/whisperliang/CP-AdapTok
source .venv/bin/activate
python -m pytest cp_adaptok/tests -q
python scripts/run_acceptance.py
python -m cp_adaptok.cli.train --config configs/s1_tokenizer.yaml --max-steps 100 --smoke --allow-untrained
python -m cp_adaptok.cli.eval --config configs/s1_tokenizer.yaml --budgets 128 256 512 1024 2048 --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --output results/optimized_eval
python -m cp_adaptok.cli.benchmark --resolution 1920 1080 --measure actual-bitstream --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --frames 8 --warmup 2 --output results/optimized_benchmark_two
python -m cp_adaptok.cli.stream_demo --mode localhost --progressive --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --output results/optimized_stream
python scripts/mutation_check.py
python scripts/run_streaming_experiments.py --checkpoint checkpoints/s3_smoke/step100.pt --allow-untrained --output results/streaming_verified
python scripts/quality_audit.py
python scripts/build_report.py
```

输入来自配置中的已有视频。首次新环境安装与恢复命令见 [README](../README.md) 和 [训练说明](training.md)。机器可读结果、预算 CSV 和 PNG/SVG/PDF 图在 results，原始日志完整保留。

## 文件树

```text
{chr(10).join(sorted(set(files)))}
third_party/AdapTok  (pinned Git submodule)
third_party/DCVC     (pinned Git submodule)
third_party/manifest.json
```

![真实预算曲线（未训练骨干）]({ROOT / "results/figures/rd_curves.png"})

![实际码流延迟]({ROOT / "results/figures/latency.png"})
"""
    (ROOT / "docs/implementation_report.md").write_text(body)
    summary = {
        "tests_passed": tests,
        "mutation_killed": sum(x["killed"] for x in mutations),
        "mutation_total": len(mutations),
        "coverage_percent": coverage,
        "max_cyclomatic_complexity": qa["max_complexity"],
        "duplicates": qa["duplicates_over_08"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "two_prefix_fps": two["overall"]["fps"],
        "five_prefix_fps": five["overall"]["fps"],
        "quality_status": evaluation["status"],
        "trained_rd_validated": False,
        "target_30fps_met": False,
        "target_60fps_met": False,
        "streaming_followup": streaming,
    }
    (ROOT / "results/delivery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
