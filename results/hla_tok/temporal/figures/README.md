# HLA-Tok-T 图表

画质均按冻结指标与同源 RGB 测量。图中 Mbps 使用 30 FPS 的应用层真实字节。定量图的原始数据与逐帧证据位于同一实验目录。PDF 保留可编辑文字，局部图像为原生帧的坐标裁剪。

| 图 | 内容 | 源数据 |
| --- | --- | --- |
| [1](figure_01_architecture.pdf) | 原生输入、HLT Record、独立恢复和双端基础引用 | [HLT v1 协议](../../../../docs/hla_tok/temporal/protocol.md) |
| [2](figure_02_correlation.pdf) | 原生 Rank 的真值、已解码和独立闭环相关性。A0 的常量预测 Pearson 未定义 | [相关性](../rank_temporal_correlation.csv) |
| [3](figure_03_energy.pdf) | 绝对表示与创新的中心化谱及熵有效维数 | [谱](../residual_spectrum.csv) |
| [4](figure_04_rate_distortion.pdf) | 同源实际 PSNR、LPIPS-Alex、DISTS–码率曲线。C0 K16320 高码率上限列于基线记录 | [带宽](../bandwidth_comparison.csv) · [C0](../baseline_reproduction.json) |
| [5](figure_05_allocation.pdf) | 四档预算下实际传输的 Rank 和分量区域比例，末 Rank 按 120 个 Region 归一化 | [分配](../rank_component_allocation.csv) |
| [6](figure_06_long_loop.pdf) | Soapbox 99 帧的重建指标与格点反量化 RMS，竖线标识 I 帧 | [逐帧](../closed_loop_per_frame.csv) |
| [7](figure_07_dynamic_bandwidth.pdf) | 真实 5→40→5 Mbps 的预算、发送应用字节与 PSNR | [闭环](../closed_loop_per_frame.csv) |
| [8](figure_08_digests.pdf) | 完整送达、增强丢失与基础丢失的双端摘要状态 | [摘要](../reference_state_audit.json) |
| [9](figure_09_bytes_used.pdf) | 符号、bitmap、包头及 CRC 的实测占用 | [账本](../rate_breakdown.csv) |
| [10](figure_10_latency.pdf) | 单卡 H100 串行 P95 与 BF16 的阶段耗时 | [延迟](../latency_breakdown.csv) |
| [11](figure_11_ablation.pdf) | 关闭或启用 Rank-aware 分配、时间预测与稀疏模式的消融。学习和运动候选按准入状态保留 | [消融](../ablation_comparison.csv) · [训练状态](../training_selection.json) |
| [12](figure_12_sample_plate.pdf) | 固定时点的原生源与局部重建，包含低运动、相机运动、纹理、长序列和源切换压力样本 | [样例身份](../samples/manifest.json) |

动态与率失真图使用确定性复现值。动态曲线的相邻点属于同一源，按帧显示，样例图的源切换行单独标注。
