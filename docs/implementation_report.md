# CP-AdapTok 实施与实验报告

可运行的工程研究原型已落地；**正式训练、有效 RD 和 1080p 30/60 FPS 验收尚未完成**。本报告直接由本地 JSON、实际测试日志生成，没有把随机骨干输出当作训练效果。

## 版本与环境

- 本报告生成前的代码提交：`b3529254b87d119cf1f2366e228985e48a4c6b16`；测量文件各自保留实际 commit、dirty 状态、source SHA256 和模型 ID。
- 选用 checkpoint：`checkpoints/s3_smoke/step100.pt`，Stage 3 结构训练 100 步，状态 `untrained_backbone_smoke`。
- checkpoint SHA256：`3aae8b811bc78e885e0905b0f15ef7aa79866dafab49837f3a20685151169eeb`。
- 模型 ID：`c663834c3bd7bf95953d6293845849f2`；codebook ID：`860d03bc14f8dde91ae57b9c8787c8ed`。
- NVIDIA H100 PCIe；torch 2.6.0；driver 580.159.03；PyTorch CUDA 12.4。关键 benchmark 串行运行，初始化/指标/I/O 不混入 pipeline FPS，状态计算没有省略。
- 官方 LD/DMCI 权重缺失；官方两个 OneDrive 分享的匿名 API 都返回 401；nvcc 和 inference_extensions_cuda 不可用。原生 Stage 0、native-parity teacher 未执行。
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

- 最终 pytest：**47 passed**；八个有意义的行为变异：**8/8 killed**。
- package 行/分支合并 coverage：84.36%。最大生产函数圈复杂度 9；AST 相似度 ≥0.8 的重复函数对数 0。
- CRAP≥10 条目数 0，细节见 `results/quality_audit.json`；不得用该审计代替未执行 native CUDA 实验。
- 本会话没有 crapper/dryer/mutator 命名工具，使用 radon+coverage、AST 检查与隔离副本行为变异作为明确记录的替代工具。详见 [操作日志](work_log.md)。
- 实际执行了所有五个要求入口：pytest 成功，正式 train/eval/benchmark/stream 的退出码分别为 [2, 2, 2, 2]，原因为缺正式权重/扩展/训练 checkpoint。显式 untrained 替代运行另行保存，未混称成功。

## 实际 1080p 质量与码率

测试为独立 BasketballDrive 短片中的 I/P 两帧；训练短片 MCL-JCV01，验证 UVG Beauty。**骨干是随机初始化，约 6.65 dB 表明质量不合格；下面仅证明工程测量链路，不能用作压缩性能结论。**

| K | PSNR dB | MS-SSIM | LPIPS | DISTS | 实际 bpp | kbps (50 FPS源) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6.65152 | 0.09413 | 1.41379 | 0.75751 | 0.00121142 | 125.6 |
| 256 | 6.65061 | 0.09414 | 1.41221 | 0.75744 | 0.00199074 | 206.4 |
| 512 | 6.65021 | 0.09413 | 1.41180 | 0.75740 | 0.00354938 | 368.0 |
| 1024 | 6.64990 | 0.09413 | 1.41183 | 0.75740 | 0.00666667 | 691.2 |
| 2048 | 6.65009 | 0.09413 | 1.41163 | 0.75738 | 0.01290123 | 1337.6 |

预算增加后实际 MSE 变差比例为 **62.5%**，没有隐瞒非单调性。当前 checkpoint 的密集/缓存重建最大像素差为 **0**，PSNR/LPIPS/DISTS 均分别记录；默认 float64 累加通过此检查。早期 fp32 的差异及失败精度证据保留于 `results/smoke100_full_eval`。

I/P 分别保存了真实 header、packet、payload 与 framing 字节；第一帧最大预算示例：`{"frame_header_bytes": 108, "packet_header_bytes": 448, "payload_bytes": 2720, "side_info_bytes": 0, "total_bytes": 3276, "transport_framing_bytes": 68}`。FrameHeader 固定108 bytes，Packet overhead 28 bytes，TCP 应用 framing 4 bytes/record。socket demo 两帧双向应用总字节 6882，包含显示 ACK；未抓取 TCP/IP/link 层开销。

## 实际码流性能

8 帧、2 帧 warmup；包含最大序列编码/全部 packet 熵编码、发送端本地基础 entropy decode/重建/状态提交、Scorer、接收端 context/entropy decode/基础和增强重建。

| 显示计划 | 整体 FPS | 稳态 P FPS | P50 ms | P95 ms | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|
| 基础+最大 | 15.06 | 17.50 | 57.15 | 105.33 | 0.984 |
| 五档显示 | 10.15 | 12.64 | 79.12 | 180.32 | 0.994 |

基础+最大时接收端每帧重建2次，另有发送端本地1次；五档时接收端5次，另有发送端1次。Attention 缓存只节省 token-to-grid 计算，后端仍按显示点重跑。不能把单个 kernel/forward 的速度当作 codec FPS。

优化前重复 CPU state SHA 读取约117.82 ms；改为设备整数摘要和 mutation-version 缓存后，实际完整 pipeline 提升到上表水平。初次摘要计算在状态提交区间内计时，后续合法读取可缓存。CPU/CUDA 摘要一致、原地修改和 inference Tensor 修改检测均有行为测试。**1080p 30 FPS 与 60 FPS 目标仍未达到**。

## 额外实验与未完成项

- 720p：真实1080p原像素的1280×720中心裁剪，明确无 resize；I/P五档实际码流已跑，不冒充独立720p数据集。见 `results/final_720_eval` / `results/optimized_benchmark_720`。
- 4096/8192：真实1080p单帧字节流、实际失真、缓存对齐已跑，状态 random_weights_structure_only。见 `results/final_extended8192`。
- 固定空间 control：独立配置与实际码流入口已跑随机结构检查，未训练，不据此作等码率优势比较。见 `results/final_spatial_control`。
- 训练：Stage 1结构100步 + resume至103；Stage 3结构100步；Stage 2/4各2步及短BPTT精确恢复测试。所有 checkpoint 均保留 untrained_backbone_smoke 状态。
- Scorer 保存真实预算标签和预测误差；在线 Fixed-K/Rate-only/Scorer 扣除已发生编码时间，基础也超时则等待；Offline Oracle 是明确非因果的全段 byte MILP。
- 未完成：官方 CUDA baseline/teacher 数值核对；正式长序列训练和全数据集 RD；训练好的空间/AdapTok同条件对照；泛化 Scorer 校准；4K；compile/CUDA Graph/fused entropy 的质量和兼容性验证；真实网络丢包/截止时间实验；物理线路开销统计。
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
python scripts/quality_audit.py
python scripts/build_report.py
```

输入来自配置中的已有视频。首次新环境安装与恢复命令见 [README](../README.md) 和 [训练说明](training.md)。机器可读结果、预算 CSV 和 PNG/SVG/PDF 图在 results，原始日志完整保留。

## 文件树

```text
configs/ablation_720.yaml
configs/ablation_8192.yaml
configs/ablation_spatial.yaml
configs/base.yaml
configs/s0_baseline.yaml
configs/s1_tokenizer.yaml
configs/s2_causal.yaml
configs/s3_entropy_scorer.yaml
configs/s4_finetune.yaml
cp_adaptok/__init__.py
cp_adaptok/backbones/__init__.py
cp_adaptok/backbones/causal_state.py
cp_adaptok/backbones/intra_adapter.py
cp_adaptok/backbones/ld_adapter.py
cp_adaptok/backbones/native_baseline.py
cp_adaptok/baselines/__init__.py
cp_adaptok/baselines/fixed_spatial.py
cp_adaptok/cli/__init__.py
cp_adaptok/cli/benchmark.py
cp_adaptok/cli/common.py
cp_adaptok/cli/eval.py
cp_adaptok/cli/stream_demo.py
cp_adaptok/cli/train.py
cp_adaptok/data.py
cp_adaptok/decoder/__init__.py
cp_adaptok/decoder/prefix_cache.py
cp_adaptok/decoder/prefix_latent_decoder.py
cp_adaptok/decoder/token_grid_projector.py
cp_adaptok/evaluation.py
cp_adaptok/metrics.py
cp_adaptok/models/__init__.py
cp_adaptok/models/causal_progressive_codec.py
cp_adaptok/runtime.py
cp_adaptok/streaming/__init__.py
cp_adaptok/streaming/bitstream.py
cp_adaptok/streaming/causal_scorer.py
cp_adaptok/streaming/entropy_codec.py
cp_adaptok/streaming/online_allocator.py
cp_adaptok/streaming/packetizer.py
cp_adaptok/streaming/receiver.py
cp_adaptok/streaming/sender.py
cp_adaptok/streaming/transport.py
cp_adaptok/tests/conftest.py
cp_adaptok/tests/test_budget_alloc.py
cp_adaptok/tests/test_causality.py
cp_adaptok/tests/test_entropy_training.py
cp_adaptok/tests/test_extended_state.py
cp_adaptok/tests/test_intra_inter.py
cp_adaptok/tests/test_packet_roundtrip.py
cp_adaptok/tests/test_prefix.py
cp_adaptok/tests/test_reference_state.py
cp_adaptok/tests/test_regression.py
cp_adaptok/tests/test_shapes.py
cp_adaptok/tests/test_state_fingerprint.py
cp_adaptok/tests/test_streaming.py
cp_adaptok/tests/test_training.py
cp_adaptok/tests/test_vq.py
cp_adaptok/tokenizer/__init__.py
cp_adaptok/tokenizer/learned_queries.py
cp_adaptok/tokenizer/lightweight_attention.py
cp_adaptok/tokenizer/prefix_masking.py
cp_adaptok/tokenizer/vector_quantizer.py
cp_adaptok/training.py
docs/architecture.md
docs/bitstream_spec.md
docs/evaluation.md
docs/implementation_report.md
docs/requirements_zh.txt
docs/training.md
docs/work_log.md
scripts/build_report.py
scripts/mutation_check.py
scripts/prepare_720p.py
scripts/quality_audit.py
scripts/run_acceptance.py
scripts/run_benchmarks.sh
scripts/run_full_eval.sh
scripts/run_smoke.sh
third_party/AdapTok  (pinned Git submodule)
third_party/DCVC     (pinned Git submodule)
third_party/manifest.json
```

![真实预算曲线（未训练骨干）](/home/whisperliang/CP-AdapTok/results/figures/rd_curves.png)

![实际码流延迟](/home/whisperliang/CP-AdapTok/results/figures/latency.png)
