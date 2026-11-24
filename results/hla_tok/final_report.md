# HLA-Tok 实验报告

## Phase 0 已通过

在 NVIDIA H100 PCIe 上评估原生 1920×1080 的 T0，Batch 1、64 张真实 Validation 帧、每点两次。模型参数 2,478,560。神经编解码单帧墙钟 FPS 使用最终代码复测。输入流水线另计已校验 JPEG 的读取与 RGB 解码、Fixed13 往返及重建回传主机。 [GPU 基准](phase0/benchmark.csv) · [流水线](phase0/pipeline/pipeline.csv)

| Token/帧 | 神经 BF16 FPS | 神经 FP32 FPS | BF16 神经 P95 毫秒 | BF16 真实输入流水线 FPS |
|---:|---:|---:|---:|---:|
| 4,096 | 133.11 | 97.20 | 7.78 | 76.82 |
| 8,192 | 91.19 | 86.21 | 11.17 | 62.85 |
| 16,320 | 69.32 | 67.90 | 14.71 | 46.63 |
| 32,640 | 47.02 | 48.73 | 22.03 | 28.98 |
| 65,280 | 28.04 | 31.50 | 36.56 | 16.60 |

K8192 BF16 的 Profiler 可计操作为 87.64 GFLOPs，选用 fused attention 等算子时 Profiler 无法覆盖所有运算。峰值 allocated 1,739 MiB、reserved 1,942 MiB，测量包含约 1,565 MiB 的预载输入和模型。BF16 的高预算计算主要受局部 Query、分块 VQ 和接收端局部注意力影响。低预算包含固定的整帧 RGB 解码成本，Fixed13 的 CPU 打包、解析和搬运也可见。本表的性能测量使用随机权重。 [分项计时](phase0/t0_final/components.csv) · [环境](phase0/t0_final/environment.json)

### 174 帧同样本基线

官方 AdapTok stride128 和 stride96 各测 FP32、BF16 六档每 Tile 时间组预算。质量覆盖 174 张真实帧。固定 16 帧窗口含 18 张尾部填充帧，对应发送索引全数计费。Joint V1 Seed 1 第 40,000 步使用 CLOSED 实发 rANS 码流重放；复核全部 RGB 重建 MSE、实发字节数及可用状态指纹，并重新计算统一 RGB 感知指标。Joint V1 表中的编码延迟来自该次归档的原始测量。 [匹配指标](baselines/metrics.csv) · [Joint V1 重放](baselines/joint_replay/metrics.csv)

| 实际测量点 | 实际 bpp | 全局 PSNR/dB | LPIPS-Alex | DISTS |
|---|---:|---:|---:|---:|
| 原版 stride128 FP32，384 Token/Tile/4 帧 | 0.089655 | 25.813 | 0.18774 | 0.08447 |
| 原版 stride128 FP32，480 Token/Tile/4 帧 | 0.112069 | 26.008 | 0.18325 | 0.08256 |
| Joint V1 CLOSED，K16320 | 0.107486 | 25.331 | 0.33248 | 0.19164 |

官方 BF16 eager 在 stride128 单预算诊断中从 Tile batch 4 的 8.12 FPS 增至 Tile batch 16 的 8.91 FPS。4 帧 KV 缓存在所测 FP32 Tile 上整数 ID 相同，RGB 最大偏差为 2.33×10⁻⁶。BF16 缓存与 Inductor 编译候选均发生整数 ID 差异，正式基线采用已核验的 eager 解码。编译后二次测量的 22.41 ms 是单 Tile 预热延迟，初次 eager 调用包含冷启动。 [Tile batch](baselines/tile_batch.csv) · [缓存等价性](baselines/cache/cache_parity.json) · [编译复测](baselines/compile_retry/official_compile.json)

### 工程和数据审计

CPU 回归 222 项通过、1 项跳过，另有 12 项不属于该次 CPU 选择。CUDA 专项 3 项通过，覆盖原生 FP32/BF16 严格整数前缀及 BF16 原生反向传播。六项定向行为变异均被测试检出。原版源码提交、官方权重和旧实验资产通过 SHA-256 及 Git 状态复核。历史三段 Test 视频的交集只读取清单。 [CPU 结果](phase0/cpu_regression_verified.txt) · [CUDA 结果](phase0/cuda_tests.txt) · [变异](phase0/mutations/summary.json) · [资产](audit/final/assets.json)

## Phase 1 未通过画质门槛

在 H100 上完成 500 步连续表示诊断和 4,500 步固定预算 VQ 训练。S2 在 256×256、512×512 和 1920×1080 依次更新 1,500、2,000 和 1,000 步，处理 2,696,192,000 个训练像素。物理及有效 Batch 均为 1。固定预算采用 40 张核验的 Train 裁块标定标准 VQ，所有 5,000 次更新的损失和梯度检查均为有限值。 [连续训练日志](phase1/continuous/updates_after_00000.jsonl) · [固定训练日志](phase1/fixed/updates_after_00000.jsonl) · [检查点 SHA-256](phase1/checkpoint_manifest.json)

| 174 帧原生 Validation | 实际 bpp | 全局 PSNR/dB ↑ | LPIPS-Alex ↓ | DISTS ↓ |
|---|---:|---:|---:|---:|
| S1 连续表示，500 步 | — | 16.245 | 0.91833 | 0.57622 |
| S2 固定 K16320，4,500 步 | 0.102315 | 16.975 | 0.74229 | 0.39796 |
| S2 同检查点绕过 VQ | — | 16.960 | 0.74290 | 0.39877 |
| 原版 stride128 FP32 实际 bpp 区间插值 | 0.102315 | 25.923 | 0.18521 | 0.08339 |
| Phase 1 验收门槛 | 0.102315 | ≥24.923 | ≤0.20373 | ≤0.09173 |

S2 通过真实 Fixed13 接收并查表，在 174 帧一共解析 4,614,480 字节的索引载荷。边界 MSE 0.020876，内部 MSE 0.020017。相同检查点绕过 VQ 后 PSNR 变化 −0.015 dB，LPIPS-Alex 变化 +0.00061，DISTS 变化 +0.00081。配对结果表明，当前大幅画质差距在量化绕过后依然存在。 [Phase 1 汇总](phase1/validation_metrics.csv) · [真实码率及阈值](phase1/fixed/fixed_quality_gate.json)

全量码字 perplexity 为 359.26，活跃码字比例为 28.39%。相邻局部 Rank 的连续表示平均余弦相似度为 0.99449，55.14% 的相邻 Rank 使用同一码字。第 4,000 至 4,500 步间，固定 9 帧验证子集的 PSNR 从 16.967 增至 17.011 dB，LPIPS-Alex 从 0.74686 升至 0.75401。质量差距、重复表示及晚期变化幅度均已记录。当前证据没有定位出可单独通过扩大 RGB 解码器容量解决的瓶颈。 [码本诊断](phase1/fixed/codebook.json) · [第 4,000 步](phase1/fixed/periodic/step_04000/validation_metrics.csv) · [第 4,500 步](phase1/fixed/periodic/step_04500/validation_metrics.csv)

原生 RGB 图和像素裁块使用 car-roundabout 第 32 帧。无损 FFV1 视频使用该序列 32–47 帧，源视频与 HLA K16320 重建同为 1920×1080、30 FPS。Git 归档包含计量与哈希，检查点、PNG 原图及 MKV 保存在实验 worktree。 [原生重建 PNG](phase1/fixed/samples/K16320.png) · [原生裁块](phase1/fixed/samples/K16320_crop.png) · [连续重建视频](phase1/video/K16320.mkv) · [视频校验](phase1/video/video_manifest.json)

## 阶段结论

Phase 0 为 PASS。Phase 1 为 FAIL，Phase 2–6 均为 BLOCKED。固定预算画质与有序表示的证据不足以进入渐进训练或启动正式长训练。前缀与因果结论的范围是无历史状态的 T0 和已验收的整数前缀。 [验收记录](acceptance.json) · [实现记录](../../docs/hla_tok/implementation_report.md)

累计读取 Train 5,040 帧、Validation 3,148 帧、Test 0 帧。读取计数包含标定、重复基准、逐周期验证、配对诊断和视频导出。官方权重、Joint V1 旧资产及原始 worktree 的 Git 状态在最终复核时与审计起点一致。 [像素读取账本](acceptance.json) · [旧资产复核](audit/final/assets.json)
