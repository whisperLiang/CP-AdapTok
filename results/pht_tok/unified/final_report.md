# PHT-Tok 双后端统一验收

**Overall Verdict 为 PARTIAL_PASS。** 统一版在官方 LD 和 HT-S 骨干上实现真实 rANS 编解码和八档可追加字节前缀。完整原生参考模式通过严格等价。当前固定最低参考层的可变预算闭环连续同步，但率失真和相对原生吞吐目标未达。

## 原生等价、渐进前缀和参考

在单卡 H100 上使用 DAVIS2017 的三段冻结 Validation，共 174 帧原生 1080p，按原顺序测 QP 16、32、48、63。LD 共 696 个帧单元，HT-S 共 104 个访问单元。完整原生参考模式的所有单元通过独立收发验证，P 单元的 Z、Y、CDF 索引、skip 掩码以及 I/P 重建和参考张量精确一致。额外使用未经 PHT 修改的官方扩展重放这些单元，码流和重建 SHA256 与双后端构建一致。 [严格等价](native_equivalence.json) [官方对照](native_stock_wire.json)

所有测量预算来自同一次完整编码。QP 32 对两个闭环后端逐单元检验八档独立解码与分段增量解码，并覆盖未完成记录和偏移重复接收。合成测试额外覆盖 Full→Minimum→Full、随机合法预算、连续最低预算、原生记忆刷新和独立 I 恢复。两端的 P 参考记忆和特征张量、I 型刷新特征逐值一致。 [前缀账本](prefix_validation.json) [状态账本](reference_validation.json)

统一后的四档 HT-S PHT1 继续支持原 MASK 和 PRECISION 格式。重放冻结 Validation 的码流字节、PSNR、MS-SSIM、LPIPS-Alex 和 DISTS，与基线逐值一致，49 个历史码流和 187 个合法前缀的 SHA256 或物理截断位置均复现。 [PHT1 兼容证据](legacy_equivalence.json)

PHT2 为 HT-S 增加八层记录组织，为 LD 接入单帧原生状态。 [原始冻结](../baseline_replay/freeze.json) [容器](backend_contract.md)

## Validation 率失真

下表为 QP 32。最低参考模式的 Base 和 Full 使用相同的最低层参考；原生参考 Full 需要先完整收齐八层。

| 后端和预算 | Mbps | Pooled PSNR dB | MS-SSIM | LPIPS-Alex | DISTS |
|---|---:|---:|---:|---:|---:|
| Native LD | 2.0249 | 30.996 | 0.93195 | 0.30814 | 0.18328 |
| PHT2 LD，原生参考 Full | 2.0539 | 30.996 | 0.93195 | 0.30814 | 0.18328 |
| PHT2 LD，最低参考 Base | 0.6179 | 20.479 | 0.64516 | 0.54931 | 0.39763 |
| PHT2 LD，最低参考 Full | 2.9112 | 29.030 | 0.89305 | 0.37709 | 0.22047 |
| Native HT-S | 1.4498 | 28.066 | 0.87816 | 0.38830 | 0.20812 |
| PHT2 HT-S，原生参考 Full | 1.4529 | 28.066 | 0.87816 | 0.38830 | 0.20812 |
| PHT2 HT-S，最低参考 Base | 0.4352 | 21.640 | 0.65409 | 0.57602 | 0.35079 |
| PHT2 HT-S，最低参考 Full | 1.5385 | 27.626 | 0.86326 | 0.41568 | 0.22581 |

QP 32 的最低完整参考成本含 I 为 LD **447,962 B**、HT-S **315,514 B**。同整数完整码流开销为 LD **1.43%**、HT-S **0.22%**。LD 在 QP 16 的开销 **4.58%** 超过 **2%** 目标，HT-S 四档均低于 **0.80%**。可变预算模式的 Full-BD-Rate 在共同 PSNR 区间为 LD **+106.9%**、HT-S **+20.5%**。以视频片段为配对单位，QP 32 的逐帧 PSNR 差按片段聚合后，三片段 bootstrap 95% 区间为 LD **[−2.39, −1.67] dB**，HT-S **[−0.58, −0.25] dB**。在各自 4,788 个逐帧相邻预算比较中，LD 的 LPIPS 和 DISTS 分别出现 5 次、42 次回退，HT-S 分别出现 13 次、24 次。 [全预算表](rate_distortion.csv) [共同区间和非单调诊断](rate_distortion_analysis.json) [物理开销](ablation.csv) [率失真图](figures/rate_distortion.pdf)

## 原始 AdapTok 对照

原 AdapTok 在同样 174 帧上完成三档 Token 预算。输入保留 1080p，使用 128×128 不重叠块，每窗 16 帧，底部反射填充至 1152；重建去掉填充并只统计真实帧。三个预算的 Pooled PSNR 分别为 22.99、25.12、26.00 dB，LPIPS-Alex 分别为 0.26246、0.20445、0.18279。固定官方 FP32 环境复核了原生编码和掩码解码。官方路径只提供 Token 预算，实际字节、Mbps、bpp 保持为空。 [AdapTok 质量](adaptok_quality.csv) [输入与逐窗记录](adaptok_validation.json)

## 运行时

使用同一段 129 帧 UVG 1080p、QP 32、FP16、单卡 H100。三次回放在原生索引 33 之后统计同一 96 帧区间。正式 Wall clock 在计时边界同步 CUDA，包含实际熵编解码和容器操作。捕获 GPU kernel 时长另外作为诊断。 [原始延迟记录](latency.csv) [吞吐及 P50/P95/P99](throughput.csv) [GPU 诊断](runtime_diagnostics.json) [运行时图](figures/runtime.pdf)

| 路径 | 串行 FPS |
|---|---:|
| Native LD | 213.51 |
| PHT2 LD Full / 逐层到达并显示 | 55.37 / 40.79 |
| Native HT-S | 282.45 |
| 原始 PHT1 HT-S Full | 171.62 |
| 统一后 PHT1 HT-S Full | 167.13 |
| PHT2 HT-S Full / 逐层到达并显示 | 146.58 / 60.65 |

PHT2 两个后端均超过 30 FPS。相对原生串行吞吐分别下降约 **74%** 和 **48%**，超过 **10%** 目标。HT-S 八帧采集等待为 UVG 120 FPS 下的 **58.33 ms**，对应 30 FPS 输入的 **233.33 ms**。所报首个显示时延包含采集等待。Native LD 的另一次 CUPTI 热身后诊断记录了 4 个 P 帧的编码 kernel，平均时长和为 2.48 ms/帧。CPU RSS 为同一运行进程的高水位；表中也保留每条路径的 CUDA 分配和保留显存样点。

## 动态传输

在相同三段 Validation 上，以真实生成的访问单元进行包级回放。五条轨迹覆盖稳定带宽、编码前变化、编码后突降、恢复、连续随机波动。两路共享同一套编码前 QP 规则、500 ms 播放缓冲、轨迹、截止时间和按时收到的反馈。PHT2 额外允许编码后选择实际存在的合法前缀，必要参考层始终计费。以 IPv4/UDP/RTP 40 B/包计算正向开销并单列反馈 48 B/单元。丢帧计 −20 dB，得分是按时可显示帧的平均贡献加上丢帧罚分。下表给出 PHT2 相对原生动态 QP 的合计得分差。离线 Oracle 是给定当前队列和已经生成码流时的可行前缀对照，服从本单元的收发、解码和参考约束。

| 带宽轨迹 | LD 在线 Δ dB | HT-S 在线 Δ dB | LD Oracle Δ dB | HT-S Oracle Δ dB |
|---|---:|---:|---:|---:|
| 稳定 | -9.47 | -0.44 | -2.90 | -0.44 |
| 编码前变化 | -15.08 | +9.86 | +7.84 | +10.19 |
| 编码后突降 | -9.16 | +1.31 | -2.06 | +2.46 |
| 恢复 | -4.13 | +8.62 | +3.28 | +14.67 |
| 连续随机波动 | -13.13 | +3.79 | +0.03 | +4.11 |

HT-S 在线前缀在 4/5 条轨迹上占优。编码后突降中，按时帧由 153/174 增至 164/174。LD 在线前缀在五条轨迹上均落后。稳定带宽下，LD 按时帧由 174/174 降至 147/174。结果支持 HT-S 在部分动态条件下从渐进截断受益。LD 的固定参考闭环代价和附加解码时延限制了同一策略的收益。数值为基于实测码流及 H100 时延的回放，三段片子的逐片配对结果、区间和实际到达字节完整保留。 [动态轨迹数据](dynamic_network.csv) [逐单元与配对区间](dynamic_evidence.json) [动态结果图](figures/dynamic_network.pdf)

## 分项裁决

| 项目 | 裁决 |
|---|---|
| Cleanup Verdict | PASS |
| Engineering Verdict | PARTIAL_PASS |
| PHT-Tok-LD Verdict | PARTIAL_PASS |
| PHT-Tok-HT-S Verdict | PARTIAL_PASS |
| Progressive RD Verdict | FAIL |
| Runtime Verdict | PARTIAL_PASS |
| Dynamic Streaming Verdict | PARTIAL_PASS |
| Overall Verdict | PARTIAL_PASS |

功能验收通过。CPU 76 项、PHT CUDA 16 项、原始 AdapTok CUDA 1 项通过。CPU 和 CUDA 定向变异共 21/21，额外 Mutator 算子子集 10/10。Ruff、mypy 通过，Dryer 在 0.8 相似度阈值下报告 0 对候选。Crapper 有 92 个函数达到或超过 10。 [复杂度报告](complexity_report.md) [测试证据](quality/summary.json)

## 研究边界

清理、冻结、代码迁移、构建恢复和测试步骤汇总在操作记录。主分支基点为 `a1c1997c2117321d26ffbedcbe2d433d560904df`。评估使用冻结 Calibration、Validation 和公开 UVG。Test 像素读取 **0**，新增优化器更新 **0**。 [清理](repository_cleanup.md) [操作记录](work_log.md) [分项裁决](final_verdict.json)
