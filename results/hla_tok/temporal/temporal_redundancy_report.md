# HLA-Tok-T 时间冗余诊断

使用 174 张原生图像构成同一视频内的 171 对真实相邻帧。A1 使用上一帧真值连续 Token，A2 使用已解包的 Q16-4 K16320 潜在表示。A3/A4 各自只使用独立接收端已提交的基础坐标。A0 使用全支持的 Rank PCA 均值作预测诊断。同一相邻对上的 RGB 指标按冻结实现计算。

| 诊断 | 预测图像 PSNR / LPIPS / DISTS | Rank Pearson 中位数 |
| --- | --- | ---: |
| A0 absolute | 13.272 / 0.83852 / 0.80584 | 未定义 |
| A1 oracle | 17.473 / 0.31960 / 0.10902 | 0.5681 |
| A2 CP decoded | 17.502 / 0.36174 / 0.15735 | 0.5695 |
| A3 copy CLOSED | 17.610 / 0.71308 / 0.48693 | 0.5562 |
| A4 linear CLOSED | 17.161 / 0.70254 / 0.45940 | 0.5247 |

按每个 Rank/Region 覆盖的原生 RGB L1 变化切分，A3 低运动残差 MSE 为 0.11433，高运动为 0.329448。低运动阈值为 2/255，高运动阈值为 8/255。

有效维数按中心化协方差特征值的归一化谱熵指数计算，每八个真实 Region 取一个样本。残差符号诊断使用统一的 I 步长。Oracle 与 CLOSED 的可用分量范围不同，差距包含基础支持和量化的影响。可部署收益由独立实际码流给出。

[相关性](rank_temporal_correlation.csv) · [分量](residual_statistics.csv) · [中心化谱](residual_spectrum.csv) · [符号频数](residual_histogram.csv) · [运动对照](motion_stratified_analysis.csv)
