# HLA-Tok V2 固定预算诊断报告

R2 在两组各 2,000 步的连续表示对照中，比 R0 提高 2.499 dB，DISTS 降低 0.10648，LPIPS-Alex 降低 0.00558。D2 的 LPIPS-Alex 门槛为降低 0.10，V2 由此结束于连续表示诊断，Phase 2 无准入资格。基于 `7c15849` 的隔离分支沿用原 K16320 质量门槛和完整 174 帧 Validation。 [架构与配对控制](../../../docs/hla_tok/v2/architecture.md)

## D0 复现及定位

三组 R0 的 174 帧结果复现通过。固定 VQ 版本在 0.102315 bpp 下为 16.975 dB、LPIPS-Alex 0.74229、DISTS 0.39796。相同权重的连续旁路为 16.960 dB、0.74290、0.39877。500 步连续版本为 16.245 dB、0.91833、0.57622。此处 bpp 以真实 Fixed13 载荷计费。 [复现数据](diag0/reconstruction_metrics.csv)

固定 VQ 检查点的 9 帧诊断中，区域公共分量占连续向量能量的 99.06%。原始相邻 Rank 余弦均值为 0.99417，去均值后的不同 Rank 余弦为 0.18094，中心化方差为 0.012175，有效秩 1.397，稳定秩 1.072。128D 输入 Query 的中心化有效秩为 59.10。实际 FP32 Q/K 诊断核验了显式 Softmax，与 SDPA 输出逐点比较。固定检查点跨 Rank 注意力图余弦均值为 0.97218，各 Rank 区间的空间覆盖接近。 [表示指标](diag0/representation_metrics.csv) · [奇异谱](diag0/figures/centered_spectrum.pdf) · [注意力热图](diag0/figures/attention_coverage.pdf)

R0 固定检查点的全量相邻同码比例为 55.14%。同权重 VQ 旁路只改变 0.015 dB，0.00061 LPIPS-Alex 和 0.00081 DISTS。该干预测量即时查表误差，量化对训练轨迹的影响需要独立控制。 [相邻同码图](figures/adjacent_same_code.pdf) · [原检查点码本统计](../phase1/fixed/codebook.json)

固定 VQ 检查点中，每区域替换为最近码字均值使全局 PSNR 降低 0.133 dB、DISTS 上升 0.1066。重复 Rank 0 使 PSNR 降低 0.188 dB。删除三组各 16 个 Rank 的全局 PSNR 变化约为 −0.002 至 −0.003 dB。打乱 Token 值和交换 Rank 对应都产生小幅指标差异。这些干预支持有序表示存在明显冗余，并保留了对区域内整体变化的敏感性。 [干预配对结果](diag0/intervention_summary.csv) · [干预图](diag0/figures/token_interventions.pdf)

两组稠密旁路都冻结 R0 空间 Encoder，共同优化输入桥接和 RGB Decoder，使用同一 500 步、131,072,000 Train 像素的 RGB 训练。128D O0 的原生诊断 PSNR 为 23.212 dB，32D O1 为 23.239 dB。相近的 O0、O1 表明冻结的空间特征在稠密条件下保留了可恢复的像素信息，两组感知重建仍有明显误差。旁路改变了空间信息数量和解码适配条件，证据不足以单独确定 Local Decoder 或 RGB Decoder 的容量上限。量化训练效应需要匹配连续控制。 [旁路指标](diag0/oracles/metrics.csv) · [定位记录](diag0/diagnosis.json)

## D1 小样本拟合

四种架构在同一 Train 单样本的 RGB-only 200 步中均通过可恢复性诊断。完整损失用固定四张 Train 裁块训练 500 步，比较共享初始化后的同一批有效像素。

| 模型 | 单样本 RGB-only PSNR/dB | 四样本 PSNR/dB | 四样本 LPIPS-Alex | 四样本 DISTS |
|---|---:|---:|---:|---:|
| R0 | 23.559 | 16.096 | 0.7373 | 0.3425 |
| R1 | 24.978 | 17.002 | 0.6124 | 0.2987 |
| R2 | 24.910 | 16.417 | 0.6709 | 0.3147 |
| R3 | 24.753 | 15.713 | 0.6910 | 0.3428 |

单样本 PSNR 相对各自初值增加 8.20–9.55 dB。R1 在四样本条件下领先，完整 Validation 的配对短程训练用于检验泛化。 [D1 数据](diag1_overfit/metrics.csv)

## D2 四组连续表示配对

Seed 1 各组均执行 1,000 次 512×512 Crop 更新，处理 262,144,000 Train 像素。公共参数和实际样本顺序的 SHA-256 一致。此表是 K16320 的 174 帧连续表示诊断。

| 模型 | 参数量 | 全局 PSNR/dB | LPIPS-Alex | DISTS | 有效秩 | 中心化方差 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 2,478,560 | 16.328 | 0.66903 | 0.55113 | 11.025 | 0.00389 |
| R1 | 2,086,144 | 16.019 | 0.77506 | 0.41633 | 2.320 | 0.64324 |
| R2 | 2,478,560 | 17.026 | 0.66948 | 0.48882 | 1.821 | 0.53346 |
| R3 | 2,611,040 | 16.885 | 1.06338 | 0.59707 | 1.947 | 0.57033 |

R2 的相邻 Rank 余弦从 R0 的 0.99891 降至 0.90668。R2 的中心化方差增大，而 R0 在很小的中心化能量下给出了更高的归一化有效秩。把区域 Token 替换成区域均值时，R2 的全局 PSNR 降低 0.908 dB，R1 降低 3.533 dB，R0 变化 +0.028 dB。直接投影和空间锚定都保留了具有重建贡献的区域内变化。仅打乱值与 Rank 对应产生的差异仍小，学习到空间变化的证据强于学到稳定 Rank 分工的证据。R1 的 DISTS 最低，R2 的 PSNR 最高，R3 的 LPIPS-Alex 退化明显。 [同样本指标](continuous_pilot/seed1/metrics.csv) · [训练曲线](figures/matched_training_curves.pdf) · [干预指标](continuous_pilot/seed1/interventions/metrics.csv)

R2 在完整 Validation 相对 R0 增加 0.698 dB，DISTS 降低 0.06231，LPIPS-Alex 增加 0.00044。首轮三项筛选同时达标的条件未满足。固定 9 帧子集的最后 250 步内，R2 PSNR 增加 0.223 dB、LPIPS-Alex 降低 0.02070、DISTS 降低 0.03552。这些走势用于触发一次 R0/R2 等量补训。 [首轮筛选](continuous_pilot/seed1/acceptance.json)

### 配对补训

两臂各从封存的第 1,000 步恢复一次，到第 2,000 步结束。每臂累计 524,288,000 Train 像素，样本链和共享初始化摘要完全一致。R0 和 R2 累计优化阶段用时分别为 163.1 和 167.6 秒，Validation 用时分别为 105.9 和 104.4 秒，训练峰值分配显存各约 5,596 MiB。曲线各测点使用同一 9 帧子集，以下数值来自完整 174 帧。 [R0 资源账本](continuous_pilot/seed1/R0/pilot/training.json) · [R2 资源账本](continuous_pilot/seed1/R2/pilot/training.json) · [配对补训曲线](figures/matched_extension_curves.pdf)

| Seed 1 连续 K16320 | 全局 PSNR/dB | LPIPS-Alex | DISTS | 中心化有效秩 | 中心化方差 | 相邻 Rank 余弦 |
|---|---:|---:|---:|---:|---:|---:|
| R0 2,000 步 | 16.628 | 0.68646 | 0.49722 | 3.507 | 0.01312 | 0.99640 |
| R2 2,000 步 | 19.127 | 0.68087 | 0.39074 | 2.861 | 1.36277 | 0.60988 |
| R2 − R0 | +2.499 | −0.00558 | −0.10648 | | | |

R2 区域公共分量能量占比为 34.88%，R0 为 99.59%。R2 与 R0 的 PSNR 增益和 DISTS 改善达到筛选要求，LPIPS-Alex 下降幅度距阈值仍差 0.09442。R2 作为本轮最有像素重建收益的锚定结构留档。该轨迹的感知质量和跨种子稳定性仍未达到架构晋级条件。 [补训对比与裁决](continuous_pilot/seed1/extension_R2/metrics.csv) · [D2 选择记录](continuous_pilot/selection.json)

补训检查点的 9 帧干预中，将 R2 的区域 Token 替换为区域均值会损失 3.266 dB，LPIPS-Alex 和 DISTS 分别增加 0.06955 和 0.09544。R0 同样操作只损失 0.035 dB。R2 区域内仅打乱 Token 值时损失 0.021 dB，表示的区域内变化有明显贡献。实际跨 Rank 注意力图的平均余弦为 R2 的 0.18858 与 R0 的 0.97532。 [配对干预](continuous_pilot/seed1/extension_R2/interventions/metrics.csv) · [跨 Rank 注意力](continuous_pilot/seed1/extension_R2/interventions/attention.csv)

## 完整 Validation 与原实际码率

| 已测模型或插值 | 174 帧实际 bpp | 全局 PSNR/dB | LPIPS-Alex | DISTS |
|---|---:|---:|---:|---:|
| Original AdapTok stride128 FP32 低端点 | 0.089655 | 25.813 | 0.18774 | 0.08447 |
| Original AdapTok stride128 FP32 高端点 | 0.112069 | 26.008 | 0.18325 | 0.08256 |
| Original AdapTok stride128 FP32 同码率线性插值 | 0.102315 | 25.923 | 0.18521 | 0.08339 |
| Original AdapTok stride96 FP32 近邻实测 | 0.097403 | 25.684 | 0.19085 | 0.08438 |
| AIP-VTok Joint V1 CLOSED K16320 | 0.107486 | 25.331 | 0.33248 | 0.19164 |
| HLA-Tok R0 Fixed13 K16320 | 0.102315 | 16.975 | 0.74229 | 0.39796 |

R0 在 0.102315 bpp 比同码率插值低 8.948 dB，LPIPS-Alex 和 DISTS 分别高 0.55709 与 0.31456。该码率下的原 Phase 1 质量条件是 PSNR 至少 24.92307 dB、LPIPS-Alex 至多 0.20373、DISTS 至多 0.09173。R2 的 19.127 dB 对应连续 K16320 诊断，其量化训练与同码率质量尚无测量。实际码率图收录已测离散点。原版 BF16 及其余码率点保存在同一基线表。 [基线全表](../baselines/metrics.csv) · [PSNR/LPIPS 与实际 bpp](figures/actual_rate_distortion.pdf) · [固定预算质量门槛](../phase1/fixed/fixed_quality_gate.json)

## H100 性能与严格前缀

H100 PCIe 使用 64 个真实原生帧、Batch 1，每档两次重复。以下是 Seed 1 第 1,000 步检查点的 BF16 FPS。Neural 包含 VQ 搜索和重建，Fixed13 增加真实载荷字节往返，Pipeline 加入源文件读取、RGB 解码和 CPU 与 GPU 之间的 RGB 传输。计时路径使用连续检查点的未校准 VQ，含通信协议的实时传输尚无实测。

| 模型 | K8192 Neural | K8192 Fixed13 | K8192 Pipeline | K16320 Neural | K16320 Fixed13 | K16320 Pipeline |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 97.39 | 73.07 | 57.32 | 73.15 | 48.51 | 40.80 |
| R1 | 114.74 | 81.98 | 65.03 | 93.76 | 56.89 | 48.16 |
| R2 | 94.98 | 71.71 | 58.15 | 70.47 | 47.44 | 40.84 |
| R3 | 86.83 | 67.03 | 51.95 | 60.24 | 42.22 | 38.24 |

两档 FP32 Neural 为 R0 87.22/69.90、R1 96.58/82.38、R2 86.82/68.89、R3 81.04/62.32 FPS。K16320 BF16 的峰值分配为 1,737–1,741 MiB，含预载的 64 帧。R3 相对 R2 每个原生帧多付持续 Rank K/V 缓存成本，K16320 BF16 Neural 降低 14.5%，且 1,000 步的三项质量均不及 R2。 [R0 性能](benchmark/seed1/R0/timings.csv) · [R1 性能](benchmark/seed1/R1/timings.csv) · [R2 性能](benchmark/seed1/R2/timings.csv) · [R3 性能](benchmark/seed1/R3/timings.csv) · [分模块时延](figures/latency_breakdown.pdf)

原生 R2/R3 八档物理前缀在 H100 的 FP32 和 BF16 均通过，涉及整帧、增量与新建独立接收端的索引一致性、字节对齐的 Fixed13 前缀和未收到后缀的隔离。CPU 回归 236 通过、1 跳过，CUDA 功能 7 通过，8 个定向变异均被检出。所有旧摘要和 Git 快照复核通过，R0 重构后的指标与旧结果相同，真实载荷字节数一致。 [CPU](cpu_tests.log) · [CUDA](cuda_tests.log) · [变异](mutations/summary.json) · [旧资产复核](legacy_assets_final.json) · [新检查点摘要](checkpoint_manifest.json)

## 原生同帧重建

冻结 Validation 的 `DAVIS2017:car-roundabout` 第 32–47 帧，16 帧、1920×1080、30 FPS、FFV1 无损 RGB。每段保存第 32 帧完整 PNG 与相同的 384×216 像素裁块，首帧 PNG 和视频解码 RGB 逐字节一致。第 1,000 步保存原始 RGB 与 R0–R3 连续同预算样例，第 2,000 步另存 R0/R2。原版 AdapTok 使用 stride128 FP32 与 480 Token/Tile/4 帧，AIP-VTok Joint V1 使用原 CLOSED 档案字节，另保留旧 R0 的实际 Fixed13 输出。原生 PNG、完整视频和权重保存在本地实验目录。 [样例索引](samples/README.md) · [逐文件校验](samples/verification.json)

## 阶段裁决

| 阶段 | 结果 | 实测范围或阻塞条件 |
|---|---|---|
| D0 | PASS | R0 三次完整重放、9 帧表示干预、两条冻结 Encoder 稠密旁路。 |
| D1 | PASS | 四臂 Train 小样本可恢复性通过。 |
| D2 | FAIL | 首轮四臂和一次 R0/R2 配对补训均未达到 LPIPS-Alex 筛选值，Seed 2 按协议未启动。 |
| D3 | BLOCKED | D2 无候选通过。已实测旧 R0 同检查点的 VQ 推理差异，V2 的配对量化训练待验证。 |
| D4 | BLOCKED | D3 无通过结果。新结构的 174 帧结果均为连续路径评估，原生帧训练未执行。 |
| D5 | BLOCKED | D4 无通过结果。功能、原生性能和旧码率基线均有实测。正式新结构离散画质验收未执行。 |

**Representation Recovery（Seed 1 诊断）** PASS。R2 的区域内能量、均值干预敏感性、PSNR 与 DISTS 均出现改进。Rank 角色分工、第二种子稳健性和离散收益仍待验证。

**Phase 1 Quality** FAIL（未获准入）。立即 VQ 误差、冻结空间特征的可恢复信息，以及锚定带来的互补性均已有直接证据。感知品质的剩余瓶颈可能涉及短训长度、Local/RGB Decoder 和损失曲线，现有对照无法进一步归因。R1 的四样本结果和时间基准支持保留直接投影作为诊断对照。若另立实验，原版 AdapTok 的同码率差距可作为 Teacher 或直接轻量化的参照。本轮按门槛结束，Phase 2 无正式训练配置。

机器验收记录保留阶段状态、实际检查点和像素计数。Test 像素读取为 0。图件均依据实测数据，8 个 PDF 通过小字和碰撞检测。曲线为单种子趋势，未估计跨种子不确定性。 [验收记录](acceptance.json) · [图源检查](figure_source_audit.json) · [实施记录](../../../docs/hla_tok/v2/implementation_report.md) · [复现命令](../../../docs/hla_tok/v2/reproduction.md)
