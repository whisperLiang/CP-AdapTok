# C0 特征冗余

128 帧原生 Train Token 用于描述分布。协方差用其中 96 帧拟合。独立校准集包含 32 帧，各视频来源与拟合集互斥。缓存包含 120 个完整空间 Rank 和含 120 个区域的 Rank 120。

全局熵有效维数为 8.66，参与率为 6.16。各 Rank 熵有效维数中位数为 9.06，范围为 5.23–9.53。累计解释方差达到 95% 和 99% 分别需要 13 和 20 维。

| 维度 | 全局解释方差 |
| --- | ---: |
| 4D | 78.77% |
| 8D | 90.10% |
| 16D | 97.17% |
| 24D | 99.93% |
| 32D | 100.00% |

六帧 RGB 消融中，平均 MSE 增幅最大的零起始分量为 identity C4, global C3, rank C0。逐分量能量与 RGB 敏感性分别可查。相邻区域水平和垂直相关性为 0.4950 和 0.4157。Rank 相关性使用 121 个 Rank 共用的前 120 个区域。

[谱](pca_spectrum.csv) · [有效维数](rank_effective_dimension.csv) · [敏感性](channel_sensitivity.csv) · [BF16 字节分布](representation_audit.json) · [协方差与相关矩阵](rank_covariance.npz)
