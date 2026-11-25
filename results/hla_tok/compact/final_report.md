# HLA-Tok-CP 最终报告

在 Seed 1 C0、原生 1920×1080、174 帧冻结 Validation 和单卡 H100 上测量。C0 Global PSNR、LPIPS-Alex 和 DISTS 复现值为 32.049857、0.094572 和 0.058787。Test 像素读取为 0。固定主 Profile 为 **Q16-4_rank / O2**。相邻帧可独立选择预算。

## 裁决

| 能力 | 裁决 |
| --- | --- |
| Experiment Completion | PASS |
| Compact Representation Correctness | PASS |
| Strict Prefix | PASS |
| Temporal Causality | PASS |
| Progressive Reconstruction | PASS |
| Equal-Payload Rate–Distortion | FAIL |
| Practical Bandwidth Quality | PARTIAL PASS |
| Runtime Performance | PASS |
| Adaptive Video Streaming Research Readiness | PARTIAL PASS |

应用端绝对画质要求为 **NOT_DECIDED**。历史 V3 Fixed13 正式裁决为 **FAIL**。 [历史裁决](../v3/final_report.md) · [复现](reproduction.json) · [历史资产](historical_assets_audit.json)

## 实测码率与画质

| Method | Profile | K | Actual bytes/frame | Total bpp | Mbps@30FPS | PSNR | LPIPS | DISTS | FPS | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 evaluated envelope | C1 Fixed13 | 16320 | 26544.0 | 0.10241 | 6.3706 | 23.25 | 0.34794 | 0.22694 | — | — |
| C0 continuous | C0 BF16 | 16320 | 1044508.0 | 4.02974 | 250.6819 | 32.05 | 0.09457 | 0.05879 | 89.31 | 11.357 |
| C0 continuous | C0 BF16 | 414 | 26524.0 | 0.10233 | 6.3658 | 14.95 | 0.54981 | 0.30835 | — | — |
| Original AdapTok | F13W, 16 frames | 2048 | 30986.5 | 0.11955 | 7.4368 | 25.99 | 0.18256 | 0.08208 | 6.61 | — |
| Compact | Q16-4 | 2596 | 20828.0 | 0.08035 | 4.9987 | 18.81 | 0.44314 | 0.24960 | 61.86 | 16.197 |
| Compact | Q16-4 | 3307 | 26516.0 | 0.10230 | 6.3638 | 19.55 | 0.41394 | 0.24317 | 61.89 | 16.205 |
| Compact | Q16-4 | 3502 | 28076.0 | 0.10832 | 6.7382 | 19.71 | 0.40376 | 0.23703 | 61.67 | 16.263 |
| Compact | Q16-4 | 5200 | 41660.0 | 0.16073 | 9.9984 | 21.89 | 0.33195 | 0.20565 | 61.44 | 16.323 |
| Compact | Q16-4 | 10409 | 83332.0 | 0.32150 | 19.9997 | 29.20 | 0.17705 | 0.12284 | 54.90 | 18.265 |
| Compact | Q16-4 | 16320 | 130620.0 | 0.50394 | 31.3488 | 29.20 | 0.17691 | 0.12267 | 59.89 | 16.742 |

Original 的 K 表示每个空间 Tile 在整个 16 帧窗口内的 Token 数。Original 以 4 帧时间 Patch 做组内处理，测量 API 等待完整 16 帧。30 FPS 输入时首帧等待 500 ms，FPS 包含物理 Fixed13 往返与拼接。C1 的带头结果使用新增的 24 字节 F13W 实验封装。 [窗口账本](original_window_ledger.csv) · [时间口径](original_temporal_scope.json) · [C1 权重](c1_reference_identity.json) · [原版归档核验](original_archive_identity.json)

## 共同字节上限

各单元格为 PSNR / LPIPS-Alex / DISTS。Best Compact 按每档联合排序，含已测的 T1 对照。固定主 Profile 的结果列于核心表。Original 按窗口均摊，使用小于或等于上限的实测档位。 [来源表](bandwidth_baseline_comparison.csv)

| 带宽 Mbps | C0 BF16 | C1 Fixed13 | Original AdapTok | Best Compact | ΔPSNR vs C0 |
| ---: | --- | --- | --- | --- | ---: |
| 5 | 14.96 / 0.5556 / 0.3189 | NOT_COMPARABLE | 25.10 / 0.2043 / 0.0924 | Q16-4 · 18.81 / 0.4431 / 0.2496 | +3.86 |
| 6.3648 | 14.95 / 0.5499 / 0.3084 | NOT_COMPARABLE | 25.80 / 0.1875 / 0.0843 | Q8-4 · T1 · 22.87 / 0.4139 / 0.2634 | +7.93 |
| 6.7392 | 15.11 / 0.5457 / 0.3055 | 23.25 / 0.3479 / 0.2269 | 25.80 / 0.1875 / 0.0843 | Q8-4 · T1 · 23.50 / 0.4002 / 0.2579 | +8.39 |
| 10 | 15.97 / 0.5199 / 0.2967 | 23.25 / 0.3479 / 0.2269 | 25.99 / 0.1826 / 0.0821 | Q8-4 · T1 · 25.73 / 0.3590 / 0.2361 | +9.77 |
| 20 | 17.17 / 0.4731 / 0.2605 | 23.25 / 0.3479 / 0.2269 | 25.99 / 0.1826 / 0.0821 | Q16-4 · 29.20 / 0.1770 / 0.1228 | +12.03 |
| 40 | 18.95 / 0.4036 / 0.2272 | 23.25 / 0.3479 / 0.2269 | 25.99 / 0.1826 / 0.0821 | Q32-4 · 30.76 / 0.1379 / 0.0979 | +11.81 |

固定主 Profile 的研究门槛。

- 5 Mbps 为 18.81 dB，门槛 23 dB。
- 10 Mbps 为 21.89 dB，门槛 26 dB。
- 20 Mbps 为 29.20 dB，门槛 28 dB。
- 40 Mbps 为 29.20 dB，门槛 30 dB。

历史 C1 的 26,520 字节对应 Payload。历史全包缺少独立账本，等总码流裁决为 **NOT_COMPARABLE**。同样本重放的 C1 F13W 包给出 26,544 字节的独立对照。 [等码率表](equal_rate_comparison.csv)

## 表示与适配

1. 全局熵有效维数 8.66，各 Rank 中位数 9.06。
2. 冻结的逐指标排序得到 16D 选用 Rank-aware PCA、8D 选用 Rank-aware PCA、4D 选用 Rank-aware PCA。
3. 纯投影损失和同 K 下额外标量损失分别见 [降维报告](projection_loss_report.md) 和 [分解表](projection_quantization_ablation.csv)。
4. K16320 的等字节高维低精度相对低维高精度结果为 Q32-4 对 Q16-8 为 +0.47 dB、Q16-4 对 Q8-8 为 +3.36 dB、Q8-4 对 Q4-8 为 +2.80 dB。感知差值和低预算见 [等比特消融](equal_bits_ablation.csv)。
5. 8D INT4 通过实际码流正确性。其联合画质低于冻结目标。实测 PSNR 为 5 Mbps 下 21.08 dB、6.3648 Mbps 下 22.50 dB、6.7392 Mbps 下 23.14 dB、10 Mbps 下 25.41 dB、20 Mbps 下 25.41 dB、40 Mbps 下 25.41 dB。
6. 5、10、20、40 Mbps 的三项质量见带宽表。
7. 主 Profile 在 26,520 字节等 Payload 下与 C1 相差 -3.68 dB，联合门槛为 FAIL。全部实测紧凑候选中，最高 PSNR 点的三项质量为 22.90 / 0.4134 / 0.2633。
8. 在 5 Mbps 共同上限下，最佳 Compact 与 Original 相差 -6.29 dB。其余带宽点见表。Original 等待 16 帧输入，比较按窗口平均字节计。
9. O1/O2 的 FP32 与 BF16 实测涵盖八档预算、50 个非预设长度及边界长度。见 [严格前缀](prefix_audit.json)。
10. 免训练主 Profile 在 40.0 Mbps 档首次用尽 K16320。实际主线从 20 到 40 Mbps 的 PSNR 增益为 +0.003 dB。
11. 主 Profile 在单卡 H100 实测 54.90–62.18 FPS，串行 P95 最高 18.265 ms。
12. T1 1000 步，T2 500 步。完整队列同码率低预算相对初始 Q8-4 获得 +0.335 dB、LPIPS 降低 0.0328、DISTS 降低 0.0128。适配后的等 Payload 为 22.90 / 0.4134 / 0.2633，保留为对照。 [适配证据](training_selection.json)。
13. 当前证据支持先改进同帧分量分配，解冻 Tokenizer 的增益尚未裁决。
14. 当前证据集中在单帧宽度与码率权衡，时间冗余压缩的增益尚未裁决。
15. 唯一推荐主线为 C。保留 Rank-aware 表示，优先用同帧实际比特预算重新分配分量，避免继续将所有 Rank 统一压到 4D。

[Train 特征谱和 RGB 消融](feature_redundancy_report.md) · [纯降维](projection_loss_report.md) · [静态尺度](scalar_calibration.csv) · [无损诊断](lossless_diagnostics.csv) · [适配](training_selection.json)

## 复现与验收

[工程验收](engineering_audit.json) · [独立接收](wire_audit.json) · [因果性](causality_audit.json) · [Profile 设备成本](profile_deployment.csv) · [H100 矩阵验收](runtime_audit.json) · [实际 H100 延迟](latency_breakdown.csv) · [完整逐帧延迟](validation_latency.csv) · [图表](figures/README.md) · [原生样例](samples/manifest.json) · [复现入口](../../../docs/hla_tok/compact/README.md)
