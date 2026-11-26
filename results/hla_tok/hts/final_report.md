# HLA-Tok-HTS 原生审计裁决

**Phase 0 = FAIL。** 单卡 H100 完成冻结 Calibration 的真实码流扫描。配置冻结所要求的低码率画质收益未出现，HT-S 集成和训练未获准入。正式 Validation 运行状态为 NOT_RUN，准入为 FAIL。Test 为 NOT_RUN，像素读取为 0。

## 低码率实测

四个原生 RGB 1920×1080 裁段共 128 帧。表中平均码率按 30 FPS 和完整物理字节计算。源清单的 FPS 为名义 30，实际采集时钟缺少独立测量。历史对齐采用 LD 32 帧和 HT 系列 33 帧 GOP。

| 模型 | 配置 | 最低实测 Mbps | PSNR dB | LPIPS | 达到 25 dB 的最低 Mbps |
|---|---|---:|---:|---:|---:|
| HT-L | matched | 0.14205 | 23.878 | 0.54204 | 0.26874 |
| HT-L | official | 0.13773 | 23.852 | 0.54393 | 0.26370 |
| HT-S | matched | 0.14905 | 23.674 | 0.54847 | 0.28829 |
| HT-S | official | 0.14422 | 23.644 | 0.55055 | 0.28096 |
| LD | matched | 0.24243 | 25.742 | 0.48218 | 0.24243 |
| LD | official | 0.22687 | 25.700 | 0.48432 | 0.22687 |

HT-S 最低达到 25 dB 的实测点为 **0.28096 Mbps**、25.173 dB、LPIPS 0.49655。配置为 official、I QP 0、P QP 12。该 LPIPS 超出相同配置的 LD 实测覆盖。

V3 A4 Base-only 在相同 Calibration 样本上为 0.37674 Mbps、26.820 dB、LPIPS 0.44160。HT-S 达到该 LPIPS 时的实测区间内插值为 0.49069 Mbps，达到相同 PSNR 时为 0.69048 Mbps。

在上述 HT-S 端点，0.486 Mbps 给每个独立 I 帧 2,025 B。最大 I 为 8,437 B，4 个 I 超限。16/20 个编码单元单独满足物理预算。严格闭环需要完整 I 参考。

## HT-S 与 LD 的等画质对照

| 配置 | 指标 | 目标 | HT-S Mbps | LD Mbps |
|---|---|---:|---:|---:|
| official | PSNR (dB) | 26 | 0.47661 | 0.25523 |
| official | LPIPS-Alex | 0.45 | 0.44900 | 0.31201 |
| official | LPIPS-Alex | 0.28978 | 2.25092 | 1.44731 |
| matched | PSNR (dB) | 26 | 0.47861 | 0.26807 |
| matched | LPIPS-Alex | 0.45 | 0.44831 | 0.32391 |
| matched | LPIPS-Alex | 0.28978 | 2.22720 | 1.44606 |

## 原生容量与正确性

| P QP | 有效 P 帧 | 不量化 y 的 PSNR 提升 dB | LPIPS 降低 | DISTS 降低 |
|---:|---:|---:|---:|---:|
| 0 | 124 | 2.127 | 0.05452 | 0.03892 |
| 12 | 124 | 2.111 | 0.05517 | 0.02947 |
| 32 | 124 | 2.087 | 0.04852 | 0.01870 |

连续诊断使用实际解码上下文与 `[1, 256, 68, 120]` 的 unquantized y。它展示官方表示的质量余量，连续路径的码率为空。

独立 I 帧、1～7 帧尾块和 32 帧序列的官方 Base 字节、RGB、参考及上下文逐项相等。未来块扰动保持历史字节和显示一致，官方权重和 buffers 的哈希相等。

## 研究与工程结论

| 问题 | 当前证据 |
|---|---|
| 1. 官方 HT-S 相对 LD 的低码率压缩 | 两套配置在 0.486 Mbps 以下的实测重叠区间均无等画质省字节端点。 |
| 2. HLA-Tok-HTS 原生 Base 保留 | 原生审计路径 parity 为 PASS。HLA-Tok-HTS 集成为 NOT_RUN。 |
| 3. 最低可接受码率 | 见上表，以 25 dB 为本次 Calibration 的报告线。 |
| 4. HLA-Tok-HTS 相对 V3 的等 LPIPS 码率 | NOT_RUN。原生对照的区间内插值见上文。 |
| 5. 渐进增强的额外率失真收益 | NOT_RUN。 |
| 6. 每增加 1 Mbps 的增强画质收益 | NOT_RUN。 |
| 7. 块级因果、前缀、增量解码 | 原生块级因果为 PASS。增强前缀和增量解码为 NOT_RUN。 |
| 8. 原生 1080p 编解码 | 选定端点原生编解码计算合计 240.01 FPS，单元 P95 28.33 ms。完整渐进运行时为 NOT_RUN。 |
| 9. 含采集与网络的实际延迟 | NOT_RUN。 |
| 10. 最终论文系统 | 当前实测支持将 HT-S 作为原生对照并报告负结果。 |

Original AdapTok 的实测码率使用独立 16 帧 F13W 窗口的有效帧平均。此前 174 帧历史 Validation 的 1.39471 Mbps 与 0.28978 LPIPS 锚点保留其原始统计范围。本轮 Calibration 曲线使用相同 128 帧，并在各自实测质量覆盖内插值。

## 归档

QP 与 I/P 组合共 408 个。逐单元账本经过原生 SPS/IP 复解析，归档 1,632 个完整原生码流。

- [原生扫点](native_audit/qp_sweep.csv)
- [低码率重叠比较](native_audit/native_comparison.json)
- [逐单元真实字节账本](native_audit/byte_ledger.jsonl)
- [旧资产与构建 SHA256](experiment_manifest.json)
- [图件清单](figures/manifest.json)
- [验证摘要](qa/summary.json)
- [执行命令和退出码](logs/command_runs.jsonl)
- [机器裁决](final_verdict.json)
