# HLA-Tok-UF-V2 单卡 H100 验收报告

**总体裁决 `PARTIAL_PASS`。** 原生 1080p 真实码流、Base-only 因果闭环、严格字节前缀和增量接收通过了实测。研究端点为 **D1 / Seed 2 final E0**。冻结率失真与等感知码率目标的完整准入未通过。Test 为 `NOT_RUN`，像素读取 **0**。分支保留在本地。

研究 checkpoint 为 `checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt`，Seed 2 累计 20,000 步，SHA-256 为 `09bb3a578d89ddb9b2e621a9cf2a1c985edaa49538f236046439116c3864c1ec`。

## 174 帧真实码流与吞吐

| 方法 | 硬上限 Mbps | 实际 Mbps | PSNR dB | LPIPS-Alex | DISTS | FPS | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anchor-Fast | 5 | 4.8870 | 26.984 | 0.46421 | 0.22755 | 57.46 | 18.29 |
| Anchor-Fast | 10 | 9.8960 | 27.920 | 0.40483 | 0.19569 | 53.63 | 19.55 |
| Anchor-Fast | 20 | 19.9157 | 29.684 | 0.28060 | 0.16677 | 47.45 | 22.05 |
| Anchor-Fast | 40 | 39.9289 | 30.650 | 0.18578 | 0.14521 | 51.55 | 20.51 |
| D1 / Seed 2 final E0 | 5 | 4.9948 | 27.350 | 0.30488 | 0.19386 | 40.92 | 25.81 |
| D1 / Seed 2 final E0 | 10 | 9.9935 | 28.058 | 0.28331 | 0.18219 | 39.09 | 26.61 |
| D1 / Seed 2 final E0 | 20 | 19.9948 | 28.646 | 0.26695 | 0.17172 | 36.11 | 29.92 |
| D1 / Seed 2 final E0 | 40 | 35.7901 | 28.862 | 0.25714 | 0.16516 | 34.72 | 30.45 |

三轮、每轮同一组 174 帧。端到端包含一次最大族真实生成、截取、独立解码/提交和显示。

[机器裁决](final_verdict.json) · [接受矩阵](evaluation/acceptance.csv) · [机器曲线](evaluation/rate_analysis.json)

## 0.5–40 Mbps 主族可达范围

| 硬上限 Mbps | 状态 | 有效帧/计划 | 实际 Mbps | PSNR dB | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|---:|
| 0 | PASS | 174/174 | 0.4379 | 26.147 | 0.51403 | 0.28304 |
| 0.5 | BASE_BUDGET_EXCEEDED | 0/174 | — | — | — | — |
| 1 | BASE_BUDGET_EXCEEDED | 0/174 | — | — | — | — |
| 1.395 | BASE_BUDGET_EXCEEDED | 32/174 | — | — | — | — |
| 2 | PASS | 174/174 | 1.9932 | 26.567 | 0.33253 | 0.21015 |
| 3 | PASS | 174/174 | 2.9939 | 26.894 | 0.31974 | 0.20272 |
| 3.594 | PASS | 174/174 | 3.5878 | 27.047 | 0.31445 | 0.19958 |
| 4 | PASS | 174/174 | 3.9942 | 27.142 | 0.31136 | 0.19782 |
| 5 | PASS | 174/174 | 4.9948 | 27.350 | 0.30488 | 0.19386 |
| 10 | PASS | 174/174 | 9.9935 | 28.058 | 0.28331 | 0.18219 |
| 20 | PASS | 174/174 | 19.9948 | 28.646 | 0.26695 | 0.17172 |
| 40 | PASS | 174/174 | 35.7901 | 28.862 | 0.25714 | 0.16516 |

0 表示完整 Base-only，采用其实际平均字节率。

## 历史基线与上游

本地方案 B 为 `3efe25ea47308cd7786888f0f5adbd0eb449e64d`，方案 A 为 `a483a212d8efeaeab045daf4f0f5d0830d69928f`，共同祖先为 `502eaf64f6fe8693dc90ddf6164c87799f1aebfe`。两个历史 worktree 的跟踪资产、冻结数据清单、评价代码、历史 checkpoint、已核对的 A/B 研究曲线及字节证据都进入只读清单。原生上游锁定 `cbdae87a5445114cdc7f48816da63ea80bdeac40`。DCVC-RT 使用方案 A 的镜像权重归档，来源局限继承该实验。A/B 均未重新训练。

Original AdapTok 的参考按 16 帧 F13W 真实应用帧窗口记账。六档账本逐组对齐 174 张有效源帧、窗口尾部填充、24 B 元数据和实际 payload。首帧等待 15 帧，30 FPS 下为 500 ms。与逐帧候选可比较实际序列 RD，逐帧硬预算和实时因果比较记 `NOT_COMPARABLE`。官方 UF-LD 与 RT 为固定 QP 的序列平均曲线。HLA-Tok-T、CC-S、CC-N、CC-J、旧 Anchor 与 D0 曲线使用同一冻结 RGB 评价，区分各模型刷新及精度。原生 I/P padding 到 1920×1088 的底部 8 行，RGB 裁回 1920×1080。

环境为 H100 PCIe、Python 3.12、PyTorch 2.9.1+cu130、CUDA 13.0。底座使用官方 FP16 CUDA Proxy 与 C++ rANS，增强恢复 FP32，D1 显示 FP16，Anchor 增强分别检验 BF16 和 FP32。纯 FP32 或 BF16 官方代理路径记 `NOT_AVAILABLE`。权重、指标和构建二进制见 [环境清单](evidence/environment.json)。全部归档基线见 [基线核验](baseline/baseline_audit.json)。

## D0 瓶颈、Anchor-Fast 与 D1

D0 历史 5 Mbps 为 6.66 FPS、P95 249.39 ms。当前 H100 未插桩的稳态 P 帧样本平均约 117 ms。带插桩的独立样本中，增强最大族组包约 127.57 ms，内含 rANS 85.14 ms、分组评分 20.84 ms、量化与 D2H 6.64 ms。平均约 3,052 次 C++ 编段、270 次解段。显示调用约 3.91 ms。用历史同步显示均值 5.95 ms 做上限估算，删除全部显示耗时后仍约 144 ms。Kineto 的 CUDA 核函数、传输和内存统计只覆盖可见活动，原生 graph 子阶段的独立时间以及精准 CPU/GPU 重叠分摊为 `NOT_AVAILABLE`。见 [阶段耗时](profiling/legacy.json) 与 [CUDA 事件](profiling/cuda_activities.json)。

Anchor-Fast 用编码端已形成的 Base 特征及官方重建头取同值预览，替代额外完整基础解码。原生批量打包避免每段 Python/C++ 往返。34 帧包含独立 I 和连续 P，Anchor 的 7,974 个边界和 D0 的 100,458 个边界均与各自旧实现逐字节一致，收端 RGB 与对照逐位相等。Anchor-Fast 四档在冻结 174 帧上保持历史 A2 质量。

D1 用相同完整原生 latent、两级有符号残差和 8×8×8 布局。I 共用原生 `dec_2` 前的 384 通道特征，P 复用原生 256 通道 feature 并保存当前帧 context。参考与显示的 Tensor 隔离。114,112 个 D1 参数约为官方 9,650,176 参数 LD 骨干的 1.18%。按对齐后原生形状统计新增卷积，I/P 精化头分别为 1.313/1.179 GMAC。见 [模型计算量](evidence/d1_compute.json)。批处理前的 34 帧同前缀对照里，5 Mbps D0-fast 显示阶段平均 5.70 ms，D1 约 3.65 ms。零步 D1 前缀语义可验证，零步质量隔离用于归因。

## 感知诊断与训练前冻结

在三个 Validation 序列顺序回放 36 帧，对 12 张原生帧执行 Q1–Q9。全量 Base / D0 5 Mbps / D0 最大合法前缀 / 连续 latent LPIPS 分别约 0.4991 / 0.4760 / 0.4090 / 0.3700。真亮度替换到 0.0466，真色度替换到 0.4186。替换高频误差到 0.0839，替换低频到 0.3339。说明亮度误差占主导，高频误差也有强效应，实际码率曲线仍受原生 latent 表示、量化和截断共同限制。排序由固定扫描的 0.4893 改善至残差排序 0.4760，一阶离线 Oracle 到 0.4698。连续与替换诊断没有作为实际压缩观测。Q0 为同口径官方归档曲线插值，未为这 12 张帧逐点求同码率固定 QP。在 [质量诊断](diagnostics/perceptual/summary.json) 和 [诊断明细](diagnostics/perceptual/frames.csv) 查逐帧 RGB/亮色/频带误差。完整诊断先于联合训练，500 步 Pilot 启动时这一组件仍在排队，与附件要求的阶段顺序有偏差。

高码率沿用原冻结规则。20 Mbps 为 28.864 dB、LPIPS 0.17480、DISTS 0.12202。40 Mbps 为 29.956 dB、LPIPS 0.12108、DISTS 0.08200。低码率为 5 Mbps 达到 25.10 dB 且 LPIPS≤0.45，10 Mbps 达到 26.00 dB 且 LPIPS≤0.30。

## 静态熵和排序

CC-S 迁移整数最大余数 CDF、批量段编码与 Raw fallback。E1 统计本模型真实 UF 符号，以帧类/精化层/通道带索引，7,296 B 整数表预装，哈希写入资产清单。同符号 E0 对照含 Base、CRC、模式字段和 rANS 终止。5/10/20/40 对应节省 44.24% / 41.31% / 42.98% / 48.24%。128 张 Calibration 帧已核对整型画布，各档 E2 实际字节与 E1 一样。合成极端符号回归命中 Raw 互转。S1 款用 Calibration 原始 MSE/LPIPS 梯度更新各通道带灵敏度。2,000 步 E1 同权重对照里，S1 在 5 Mbps LPIPS 约 0.34117，S0 为 0.34076，10 Mbps 为 0.32853 与 0.32880。没有稳定净收益，保留 S0。见 [同符号账本](entropy/same_symbol.csv) 与 [排序消融](figures/fig18_perceptual_ordering.pdf)。

## Base-Lite 与低码率

Base-Standard 固定 I=0/P=8，冻结 174 帧 I/P 峰值 7,593 / 6,322 B，Calibration 峰值 8,563 / 6,846 B，均低于 9,212 B。官方最低 QP 的 Base-Lite 为 I=0/P=0，Validation 均值 0.2838 Mbps，I/P 峰值 7,593 / 4,290 B。6 个 I 与 1 个 P 超过 4,096 B。独立校准试了五组固定原生量化步长和 P 分析增益，最低 I 峰值仍为 8,016 B。没有配置通过 Calibration，附加 Base-Lite 候选未进入 Validation。每个候选每帧只执行一次物理编码。见 [最小 QP](diagnostics/base-validation/summary.json) 与 [扩展校准](diagnostics/base-lite-extended/summary.json)。

0.5/1/1.395 Mbps 的 Base-Standard 主族在冻结序列中遇到基础层不可达，记录 `BASE_BUDGET_EXCEEDED`，后继依赖帧记录 DESYNC 直至下一完整 I。2/3/3.594/4/5/10/20/40 Mbps 对照只在完整合法前缀上计算全量质量。原生 y/z 共用 rANS 载荷，全部字节计入，独立 Hyperprior 字节不可辨识。实验没有网络封包/FEC/链路 wire bytes 测量。

## 状态、前缀、增量和动态带宽

统一应用层访问单元含 30 B Base 头和 12 B 增强段 (header 与 CRC)。增量只累加新符号和当前组的合法层级，半段不解码，`close()` 标明访问单元结束。完整 Base 只提交一次，显示 `render()` 才执行共享读出。损坏段、缺依赖和重复段有负测，CRC 通过的错误模型 tag 仍会被拒绝。相同材质的独立收端能逐段验证符号和 Base。动态带宽包括 5→40→10→20→5，以及 Base 可达时的 1/2/3 Mbps。真实 99 和 90 帧窗口合计 189 帧，动态与 Base-only 状态失配 0。两次丢 Base 后拒收 38 个依赖 P，独立 I 恢复 2 次。128/256 连续帧不存在匹配冻结素材。见 [动态证据](evaluation/dynamic-static/summary.json)。

## 图表与复现

[图 1 架构](figures/fig01_architecture.pdf) · [图 2 解码结构](figures/fig02_decoders.pdf) · [图 3 D0 热点](figures/fig03_d0_hotspots.pdf) · [图 4 Anchor 延迟](figures/fig04_anchor_latency.pdf) · [图 5 D0/D1 吞吐](figures/fig05_d0_d1_speed.pdf)

[图 6 PSNR](figures/fig06_psnr_rd.pdf) · [图 7 LPIPS](figures/fig07_lpips_rd.pdf) · [图 8 DISTS](figures/fig08_dists_rd.pdf) · [图 9 等 PSNR](figures/fig09_equal_psnr.pdf) · [图 10 等 LPIPS](figures/fig10_equal_lpips.pdf)

[图 11 Base 字节](figures/fig11_base_bytes.pdf) · [图 12 静态熵](figures/fig12_cc_s_savings.pdf) · [图 13 开销拆分](figures/fig13_application_byte_breakdown.pdf) · [图 14 感知诊断](figures/fig14_perceptual_diagnostics.pdf) · [图 15 原生内容](figures/fig15_native_comparison.pdf)

[图 16 原生裁块](figures/fig16_native_crop.pdf) · [图 17 训练曲线](figures/fig17_training.pdf) · [图 18 排序](figures/fig18_perceptual_ordering.pdf) · [图 19 闭环](figures/fig19_causal_reference.pdf) · [图 20 质量—速率—吞吐](figures/fig20_quality_rate_throughput.pdf)

图中性能是技术重复，三次复跑共 522 帧/档，平均 FPS 和全帧 P95。率失真是单 checkpoint 的 174 帧汇总。相同前缀增益按同源帧配对，未把软单调损失当成数学保证。图 15 的显示版缩放只用于排版，图 16 从同一原生输入坐标裁出。码流和预览的来源与 SHA-256 保存在 [样例清单](samples/native/manifest.json)。

20 张图均有 PDF、SVG 和 600 dpi TIFF。源脚本、最小字号、PDF 碰撞和多面板对齐经复核。见 [图表 QA](evidence/figure_qa.json)。

[复现命令](../../../docs/hla_tok/uf_v2/reproduction.md) · [性能口径](../../../docs/hla_tok/uf_v2/performance_audit.md) · [工作日志](../../../docs/hla_tok/uf_v2/work_log.md)

## 单卡 H100 训练与续训

| Seed | 阶段 | 新增更新 | 累计节点 | 展开帧 | GPU h | 样本帧次 | 峰值分配 GiB | 平均 s/步 |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | P1 | 2,000 | 500 / 2000 | 4 / 8 | 0.02307 | 14,000 | 29.25 | 0.0215 |
| 1 | P2 | 6,000 | 4000 / 6000 / 8000 | 8 | 0.07916 | 48,000 | 13.94 | 0.0388 |
| 1 | P3 | 12,000 | 10000 / 12000 / 14000 / 16000 / 18000 / 20000 | 16 | 0.21867 | 192,000 | 15.27 | 0.0569 |
| 2 | P1 | 2,000 | 500 / 2000 | 4 / 8 | 0.02336 | 14,000 | 29.20 | 0.0224 |
| 2 | P2 | 6,000 | 4000 / 6000 / 8000 | 8 | 0.07802 | 48,000 | 13.91 | 0.0383 |
| 2 | P3 | 12,000 | 10000 / 12000 / 14000 / 16000 / 18000 / 20000 | 16 | 0.21753 | 192,000 | 15.27 | 0.0565 |

每次更新采样一个连续窗口，梯度累积 1。RGB 裁块为 128×192，占原生画面 1.19%。训练缓存来自真实闭环，P2/P3 在每个节点重建缓存并停止历史梯度。GPU 小时包括训练缓存与实际更新，验证与性能复测单列。每步耗时、梯度与估计 bpp/缓存实际 bpp 保留在逐步日志。官方模型与历史 A/B 权重沿用归档，官方预训练成本为 `NOT_AVAILABLE`。

Seed 1 正式阶段结束于累计 20000 步，结束条件为 达到正式累计上限。8,000 步与最佳 20000 步的低档综合 RD 为 0.041478 和 0.039096。 [逐节点停止规则证据](training/formal-seed1/continuation.json)

Seed 2 正式阶段结束于累计 20000 步，结束条件为 达到正式累计上限。8,000 步与最佳 20000 步的低档综合 RD 为 0.041841 和 0.039201。 [逐节点停止规则证据](training/formal-seed2/continuation.json)

Seed 1 末个正式训练节点的平均率代理为 0.1641 bpp，闭环缓存的真实率为 0.2191 bpp。部署表单列物理字节。

Seed 2 末个正式训练节点的平均率代理为 0.1636 bpp，闭环缓存的真实率为 0.2196 bpp。部署表单列物理字节。

历史 Anchor A2 两个种子的训练计时为 78.68 s 和 78.87 s，沿用已完成的权重。 [历史 A2 训练账本](training/inherited_anchor_ledger.json)

另有 1 步 P2 冒烟和缓存复现，耗时 0.00571 GPU h。P1、P2、P3 断点恢复均核对原 checkpoint 的模型权重、优化器、随机状态、数据身份和全局更新数。固定 LR 存于冻结配置。checkpoint 内含配置与训练源哈希，原生 Gitlink 和概率表哈希使用与 checkpoint SHA-256 绑定的验证证据。

[优化器账本](training/optimizer_ledger.json) · [Seed 2 准入](training/seed2_admission.json)

## D0 与蒸馏后 D1 的同前缀对照

冻结 174 帧逐一重生验证，共 492,324 个边界。两个模型各自带独立标识。原生 Base payload、增强字节、所有合法结束位置和 Base 帧元数据完全相同。 [物理配对证据](evidence/paired-d0-d1.json)

| 帧上限 Mbps | D0 PSNR | D1 PSNR | D0 LPIPS | D1 LPIPS | D0 DISTS | D1 DISTS |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 27.07807 | 26.76643 | 0.48975 | 0.35124 | 0.26476 | 0.21657 |
| 10 | 27.57460 | 27.14505 | 0.47428 | 0.33825 | 0.25741 | 0.21161 |
| 20 | 28.23631 | 27.60030 | 0.45179 | 0.32394 | 0.24647 | 0.20278 |
| 40 | 28.63447 | 27.77336 | 0.43255 | 0.31433 | 0.23894 | 0.19690 |

P2/P3 联合更新会改变最终 Base 和增强符号。最终结果用相同硬预算和实际码率比较。

## 冻结配置的熵、排序与高频对照

| 配置 | LPIPS @5 | LPIPS @10 | LPIPS @20 | LPIPS @40 |
|---|---:|---:|---:|---:|
| D1 / fixed | 0.40299 | 0.38645 | 0.35167 | 0.31821 |
| D1 / fixed static | 0.36614 | 0.33182 | 0.31658 | 0.31357 |
| D1 / 2000 | 0.35124 | 0.33825 | 0.32394 | 0.31433 |
| D1 / static | 0.34076 | 0.32880 | 0.31491 | 0.31357 |
| D1 / perceptual | 0.34117 | 0.32853 | 0.31483 | 0.31357 |
| D1 / L0 4000 | 0.34117 | 0.32774 | 0.31313 | 0.30325 |
| D1 / joint 4000 | 0.34201 | 0.32836 | 0.31367 | 0.30381 |
| D1 / formal E0 fast | 0.30827 | 0.28506 | 0.26905 | 0.25859 |
| D1 / formal static fast | 0.31219 | 0.28787 | 0.26992 | 0.25850 |

高频对照固定同一个 Seed 1 的 2,000 步起点、采样位置、冻结超参数和 174 帧评价。L0 单独将高频系数置 0，追加 2,000 步和 16,000 帧次，耗时 0.02610 GPU h。该次控制中 L1 的 PSNR 高 0.005–0.010 dB，LPIPS 也高 0.0005–0.0009。DISTS 在四档有升有降。 [高频对照训练账本](training/l0-seed1/step-4000.json)

## 端到端分项与检测

| Mbps | 编码 FPS | 解码 FPS | Base 可用均值 ms | P50 ms | P99 ms | 稳态 FPS | 三轮 FPS 范围 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 68.20 | 103.25 | 19.23 | 24.27 | 38.63 | 41.30 | 40.32–41.86 |
| 10 | 68.61 | 91.62 | 19.15 | 25.07 | 40.01 | 39.46 | 39.00–39.25 |
| 20 | 67.61 | 78.09 | 19.37 | 27.21 | 42.72 | 36.38 | 35.68–36.49 |
| 40 | 68.11 | 71.30 | 19.27 | 28.61 | 43.98 | 34.98 | 34.60–34.95 |

计时峰值 CUDA 分配 4.23 GiB。完整 I/P、各轮及已预热分布保存在逐档账本。60 FPS / P95 16.67 ms 优化目标尚未达到。 [所选码流计时](d1/formal-seed2-final-e0/summary.json)

Anchor-Fast FP32 增强 的 40 Mbps 为 52.60 FPS、P95 20.08 ms。样本 522 帧含三轮。 [计时账本](anchor_fast/fp32/summary.json)

Anchor-Fast 按需 的 40 Mbps 为 50.73 FPS、P95 33.17 ms。样本 102 帧含三轮。 [计时账本](anchor_fast/on-demand/summary.json)

D1 E1 优化前按需 的 40 Mbps 为 29.79 FPS、P95 54.83 ms。样本 102 帧含三轮。 [计时账本](d1/on-demand/summary.json)

Seed 1 E0 优化后按需 的 40 Mbps 为 34.94 FPS、P95 40.86 ms。样本 102 帧含三轮。 [计时账本](d1/selected-on-demand/summary.json)

Seed 2 E0 优化后按需 的 40 Mbps 为 34.95 FPS、P95 42.17 ms。样本 102 帧含三轮。 [计时账本](d1/selected-on-demand-seed2/summary.json)

独立 CPU 输入测量读取 174 张 Validation，含路径检查和哈希的文件读取 P50 0.84 ms，含图像解码的总 P50 11.54 ms。系统缓存由实际运转决定。网络为 `NOT_RUN`。 [CPU 输入账本](profiling/inputs/summary.json)

## 最终码流重放

所选码流 tag 为 `c67eee930aa1446e`。动态回放 189 帧，Base-only DPB 失配 0，原生 Base 最大 7,566 B。两次原生 I 恢复，丢 Base 后拒收 38 个依赖 P。 [闭环和概率表哈希](evaluation/dynamic-seed2-final/summary.json)

在四个原生访问单元逐条追加 16,534 个增强记录，整帧 RGB 与一次收完逐位完全相同。每条追加 P50 8.46 µs，P95 10.79 µs。首次 Base 可用与每次/累计重渲染见逐帧账本。 [逐条追加延迟](evaluation/incremental-seed2-final/summary.json)

独立插桩构建对两个真实 P 帧量得 C++ 整数画布写入 P95 1.038 µs 和 0.834 µs。每次逐记录恢复均与批量整数画布一致。 [整数写入计时](profiling/incremental_canvas_seed2.json)

[原生样例及字节 SHA-256](samples/native-seed2-final/manifest.json)

## 研究端点选择

| 候选 | 四档物理预算 | 四档实时性 | 低档质量通过 | 高档质量通过 | 等 LPIPS 0.28978 实际 Mbps |
|---|---:|---:|---:|---:|---:|
| Anchor-Fast | 4/4 | 4/4 | 0/2 | 0/2 | 18.9123 |
| D1 / formal static fast | 4/4 | 4/4 | 2/2 | 0/2 | 9.4635 |
| D1 / formal E0 fast | 4/4 | 4/4 | 2/2 | 0/2 | 8.6806 |
| D1 / Seed 2 final E0 | 4/4 | 4/4 | 2/2 | 0/2 | 8.1173 |

研究端点为 **D1 / Seed 2 final E0**。在通过物理预算、实时性、闭环和低档质量的候选中，按同感知目标的实际传输量比较。Anchor-Fast 的 5/10 Mbps 感知质量落后于 D1，20/40 Mbps 的 PSNR 和吞吐更高，40 Mbps 的 LPIPS 与 DISTS 也更低。静态 E1 在 Seed 1 的 40 Mbps 最大族保留近似相同画质，实际码率由 35.8629 降到 31.0474 Mbps，较低硬预算上的感知结果使其在主目标中排名较后。全部候选均未满足部署研究门槛。

## 逐序列与相邻前缀

| 硬上限 Mbps | 指标 | 退化帧 | 退化比例 | 最差退化 |
|---|---|---:|---:|---:|
| 5 → 10 | PSNR (dB) | 0/174 | 0.00% | 0.000000 |
| 5 → 10 | LPIPS | 0/174 | 0.00% | 0.000000 |
| 5 → 10 | DISTS | 1/174 | 0.57% | 0.000917 |
| 10 → 20 | PSNR (dB) | 0/174 | 0.00% | 0.000000 |
| 10 → 20 | LPIPS | 0/174 | 0.00% | 0.000000 |
| 10 → 20 | DISTS | 2/174 | 1.15% | 0.001613 |
| 20 → 40 | PSNR (dB) | 0/174 | 0.00% | 0.000000 |
| 20 → 40 | LPIPS | 1/174 | 0.57% | 0.000636 |
| 20 → 40 | DISTS | 2/174 | 1.15% | 0.000925 |

逐帧变化的绝对量按冻结阈值 10⁻⁷ 计数。 [逐序列聚合](d1/formal-seed2-final-e0/by_video.csv) · [逐帧质量](d1/formal-seed2-final-e0/quality.csv) · [各序列最差退化](d1/formal-seed2-final-e0/adjacency.csv)

| 帧上限 Mbps | 实际 Mbps | ΔPSNR dB | ΔLPIPS | ΔDISTS |
|---:|---:|---:|---:|---:|
| 5 | 4.9948 | -6.8422 | +0.1014 | +0.0619 |
| 10 | 9.9935 | -8.8785 | +0.1479 | +0.0872 |
| 20 | 19.9948 | — | — | — |
| 40 | 35.7901 | — | — | — |

Δ 为候选减去同实际序列码率下的官方 UF‑LD。共同区间外保留空值。 [官方 UF-LD 逐档差值](evaluation/official_ld_penalty.csv)

## 回归、变异和封存

CPU 全仓回归 462 通过、1 跳过。CUDA 环境的历史 UF 和 UF‑V2 两套回归共 75 通过。定向变异 16 项 CPU、3 项 CUDA，全部被测试捕获。ub-mode Mutator 独立抽样 6 个站点全部捕获，与定向语义集合存在重叠。Ruff 与 mypy 通过。Crapper 对 UF‑V2 Python 运行时最大 CRAP 7.0，Dryer 在该范围的 0.80 阈值命中 0 项。实验脚本发现两种 checkpoint 原子保存的结构相似度 0.80，保留冻结训练源码哈希。

Test 像素读取 0。历史 A/B 分支及其完整资产摘要复核一致。

[回归和工具证据](evidence/qa_summary.json) · [40 项合同](evidence/contracts.json) · [Test 审计](evidence/test_audit.json)



## 等画质比较与最终裁决

| 目标 | D1 实际 Mbps | Original AdapTok 实际 Mbps | 裁决 |
|---|---:|---:|---|
| 25 dB | — | 3.5935 | OUT_OF_RANGE |
| 26 dB | — | — | OUT_OF_RANGE |
| 28 dB | 9.4390 | — | OUT_OF_RANGE |
| 30 dB | — | — | OUT_OF_RANGE |
| LPIPS 0.45 | 0.7474 | — | OUT_OF_RANGE |
| LPIPS 0.30 | 5.8428 | — | OUT_OF_RANGE |
| LPIPS 0.28978 | 8.1173 | 1.3947 | FAIL |
| DISTS 0.15 | — | — | OUT_OF_RANGE |
| DISTS 0.20 | 3.5019 | — | OUT_OF_RANGE |

在硬上限 2 Mbps，完整序列实测 26.567 dB、1.9932 Mbps，满足观察到 ≥25 dB 的阈值。 精确 25 dB 只在共同 RD 覆盖内插值。Original 的 0.28978 来自实测值 0.2897786209466814 四舍五入，比较按原始有效点计算。

[等画质表](evaluation/equal_quality.csv) · [BD-Rate 与公共积分区间](evaluation/bd_rate.csv) · [官方 UF-LD 代价](evaluation/official_ld_penalty.csv)

| 维度 | 状态 |
|---|---|
| BASELINE | `PARTIAL_PASS` |
| PROFILING | `PARTIAL_PASS` |
| ANCHOR_FAST | `PASS` |
| D1_DECODER | `PASS` |
| UNIFIED_LATENT | `PASS` |
| STATIC_ENTROPY | `PASS` |
| BASE_LITE | `FAIL` |
| PERCEPTUAL_ALLOCATION | `PARTIAL_PASS` |
| RATE_DISTORTION | `FAIL` |
| EQUAL_PSNR_RATE | `OUT_OF_RANGE` |
| EQUAL_PERCEPTUAL_RATE | `FAIL` |
| STRICT_PREFIX | `PASS` |
| CAUSALITY | `PASS` |
| BASE_BUDGET | `PASS` |
| INCREMENTAL | `PASS` |
| REALTIME | `PASS` |
| TRAINING | `PASS` |
| TEST | `NOT_RUN` |
