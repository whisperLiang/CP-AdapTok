# HLA-Tok-T 最终报告

冻结 C0 原生 1920×1080、174 帧 Validation。选出的实测 Temporal 候选为 **T0-RA S2**，基础层 **B2**，16320 个固定基础分量、64 条基础 Record，最坏完整基础 9212 字节。
使用 4-bit、256 分量短块和 32 帧刷新间隔。Test 像素读取 **0**。

## 裁决

| 项目 | 结果 |
| --- | --- |
| Experiment Completion | PARTIAL PASS |
| Baseline Reproduction | PASS |
| Native 1080p | PASS |
| Temporal Redundancy Benefit | FAIL |
| Strict Temporal Causality | PASS |
| Strict Integer Prefix | PASS |
| Strict Wire Prefix | PASS |
| Incremental Reconstruction | PASS |
| Base Reference Synchronization | PASS |
| Enhancement Independence | PASS |
| Long-Horizon Closed Loop | PASS |
| Base Loss Recovery | PASS |
| Equal-Payload Rate–Distortion | FAIL |
| Practical Bandwidth Quality | PARTIAL PASS |
| Runtime Performance | PASS |
| Dynamic Bandwidth Adaptation | PASS |
| Research Readiness | PARTIAL PASS |

## 真实码流与画质

质量顺序为 Global PSNR、LPIPS-Alex、DISTS。码率按应用字节在 30 FPS 下换算。Original 使用同一 174 帧的已核验 Fixed13 归档，按 16 帧窗口将尾窗填充成本均摊到真实源帧。原版首帧等待为 15 帧，即 30 FPS 下 500 ms。HLT 以完整单帧作为输入，算法前视为 0 帧。

| 上限 Mbps | HLT B/frame | 实际 Mbps | HLT 三项质量 | CP Q16-4 | CP Q8-4 T1 | Intra RA | Original AdapTok |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 5 | 20764.8 | 4.9836 | 22.061 / 0.56366 / 0.36357 | 18.814 / 0.44314 / 0.24960 | 21.358 / 0.45084 / 0.28131 | 21.068 / 0.63425 / 0.41569 | 25.105 / 0.20427 / 0.09239 |
| 10 | 41595.3 | 9.9829 | 25.247 / 0.39092 / 0.25175 | 21.888 / 0.33195 / 0.20565 | 25.735 / 0.35900 / 0.23608 | 23.147 / 0.50831 / 0.32124 | 25.996 / 0.18302 / 0.08236 |
| 20 | 83261.3 | 19.9827 | 28.864 / 0.17480 / 0.12202 | 29.199 / 0.17705 / 0.12284 | 25.742 / 0.35913 / 0.23577 | 27.662 / 0.25919 / 0.17891 | 25.996 / 0.18302 / 0.08236 |
| 40 | 166598.3 | 39.9836 | 29.956 / 0.12108 / 0.08200 | 29.202 / 0.17691 / 0.12267 | 25.742 / 0.35913 / 0.23577 | 30.385 / 0.12963 / 0.08732 | 25.996 / 0.18302 / 0.08236 |

相对差值按 HLT 减去对照计算。感知指标下降对应改善。

| 上限 Mbps | 对 CP Q16-4 | 对 CP Q8-4 T1 | 对 Intra RA | 对 Original |
| ---: | --- | --- | --- | --- |
| 5 | +3.25 / +0.1205 / +0.1140 | +0.70 / +0.1128 / +0.0823 | +0.99 / -0.0706 / -0.0521 | -3.04 / +0.3594 / +0.2712 |
| 10 | +3.36 / +0.0590 / +0.0461 | -0.49 / +0.0319 / +0.0157 | +2.10 / -0.1174 / -0.0695 | -0.75 / +0.2079 / +0.1694 |
| 20 | -0.34 / -0.0023 / -0.0008 | +3.12 / -0.1843 / -0.1138 | +1.20 / -0.0844 / -0.0569 | +2.87 / -0.0082 / +0.0397 |
| 40 | +0.75 / -0.0558 / -0.0407 | +4.21 / -0.2380 / -0.1538 | -0.43 / -0.0085 / -0.0053 | +3.96 / -0.0619 / -0.0004 |

共同 26,520 B Payload 条件下，T0-RA S2 为 23.435 / 0.49239 / 0.31313，C1 Fixed13 为 23.250 / 0.34794 / 0.22694。HLT 实际总字节均值为 29765.8 B/frame。
同总字节 26,544 B 及 6.7392 Mbps 的结果见 [共同上限](equal_payload_comparison.csv) 与 [完整带宽表](bandwidth_comparison.csv)。逐帧硬上限包含 I 帧、协议头、CRC、bitmap 和补位。

## 贡献与训练准入

| 对照 | 5 Mbps ΔPSNR / ΔLPIPS / ΔDISTS | 10 Mbps ΔPSNR / ΔLPIPS / ΔDISTS |
| --- | --- | --- |
| 时间预测与实际稀疏前缀，S0 | -0.01 / -0.0053 / -0.0024 | +0.01 / -0.0036 / -0.0024 |
| 时间预测与实际稀疏前缀，锁定 S1 | +0.03 / -0.0001 / +0.0001 | +0.09 / -0.0065 / -0.0023 |
| Rank-aware 分配，绝对编码 | +0.06 / +0.0666 / +0.0642 | +0.46 / +0.0664 / +0.0593 |
| 能量分量分配 S1，复制闭环 | +0.09 / +0.0718 / +0.0668 | +0.54 / +0.0634 / +0.0594 |
| RGB/实际比特分量分配 S2，复制闭环 | +0.96 / -0.0705 / -0.0522 | +2.01 / -0.1109 / -0.0672 |
| 稀疏短块，锁定复制及 S1 | +0.06 / -0.0013 / -0.0020 | +0.21 / -0.0093 / -0.0043 |
| 稀疏短块，锁定复制及 S0 | +0.07 / -0.0065 / -0.0039 | +0.12 / -0.0050 / -0.0038 |
| 有界外推相对历史复制 | -0.12 / +0.0055 / +0.0026 | -0.18 / +0.0035 / +0.0024 |

准入比较使用每个码率下绝对 HLT 和历史 CP 对照的逐指标最强值。逐来源成对结果见候选记录。

- 5 Mbps。相对最强对照 ΔPSNR -0.344 dB，LPIPS 降幅 -0.12052，DISTS 增幅 +0.11397。0/3 个来源过门槛。
- 10 Mbps。相对最强对照 ΔPSNR -0.488 dB，LPIPS 降幅 -0.05897，DISTS 增幅 +0.04610。0/3 个来源过门槛。
- 20 Mbps。相对最强对照 ΔPSNR -0.335 dB，LPIPS 降幅 -0.00031，DISTS 增幅 +0.00182。2/3 个来源过门槛。

锁定 S1 的 5 Mbps 对照中，绝对基础 Payload 为 8160.0 B/frame，复制闭环为 7407.6 B/frame。同档位 T0-RA 稀疏发送 38599.4 个分量/帧，RAW 发送 37056.0。较小的基础 Payload 释放了增强容量，三项画质仍由整条扫描及有效表示决定。


Phase 3 和 Phase 4 的训练状态分别为 NOT_RUN 与 NOT_RUN。实际优化器更新 0 步。T0 与有界线性外推在相同字节运行。T1 与 T2 的状态由前序准入裁决保留。

固定 B2 支持的 48 对补充诊断中，Oracle 相对 CLOSED 的差值为 +0.007 dB / -0.00235 LPIPS / -0.00356 DISTS。完整支持 Oracle 的较大差距包含未覆盖分量的贡献。 [同支持诊断](matched_support_oracle.csv)

提高预算后的逐帧画质下降次数为 PSNR 0、LPIPS 0、DISTS 0，每项检查 522 个相邻预算对。

## 参考状态与 H100

在 64、90、99 帧真实连续片段上完成 2,783 个帧条件。完整基础送达时摘要不一致 0 次。第 7 帧基础 Record 丢失后，第 32 帧的下一次周期 I 刷新在三个视频上恢复 3 次。Soapbox 固定 10 Mbps 的前、中、后段 PSNR 为 26.56 / 26.49 / 26.79 dB。重复静止帧的整数基础收敛通过。冻结 Validation 的 128 和 256 帧连续条件为 NOT_AVAILABLE。

| 精度 | Mbps | HLT FPS | HLT P95 ms | 相对 CP ΔFPS | 相对 CP ΔP95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 5 | 49.57 | 20.34 | -12.29 | +4.15 |
| BF16 | 10 | 45.94 | 21.91 | -15.50 | +5.59 |
| BF16 | 20 | 41.00 | 24.58 | -13.90 | +6.31 |
| BF16 | 40 | 33.92 | 29.78 | -25.97 | +13.04 |
| FP32 | 5 | 50.11 | 20.09 | -12.05 | +3.79 |
| FP32 | 10 | 46.53 | 21.63 | -15.63 | +5.50 |
| FP32 | 20 | 41.62 | 24.21 | -14.71 | +6.30 |
| FP32 | 40 | 34.23 | 29.50 | -26.43 | +12.98 |

参考 Python/NumPy Record 路径的 BF16 40 Mbps 为 27.27 FPS、P95 36.83 ms。采用 C11 Record 后端前在实际帧上完成 512 次码流、账本、整数显示和状态摘要一致性检查。C11 按批融合短块打包及封装、解析及短块恢复。各融合阶段按实际调用计时。 [Python 路径原测量](runtime_audit_python.json) · [后端等价审计](native_wire_parity.json)

H100 延迟以 GPU 原生单帧输入至同步后的 GPU 重建计。输入文件读取和实际网络传输各自独立。CP 性能对照使用原归档代表帧重复口径，Temporal 在真实片段边界重置参考后循环。

## 工程证据

工程审计为 **PASS**，历史资产审计为 **PASS**。接收端在独立进程用实际 HLT 字节和冻结合成权重生成原生 RGB。

[基线复现](baseline_reproduction.json) · [时间冗余](temporal_redundancy_report.md) · [选型](candidate_selection.json) · [闭环明细](closed_loop_per_frame.csv) · [动态带宽](dynamic_bandwidth_results.csv) · [字节分解](rate_breakdown.csv) · [延迟](latency_breakdown.csv) · [工程审计](engineering_audit.json) · [Test 审计](test_access_audit.json) · [图表](figures/README.md) · [样例](samples/manifest.json) · [复现](../../../docs/hla_tok/temporal/README.md)

下一步优先在固定 5/10 Mbps 口径下重配基础层的 Rank×Component 覆盖，先让参考层覆盖对 RGB 和感知质量更重要的分量，再重新测量时间预测的同字节净收益。
