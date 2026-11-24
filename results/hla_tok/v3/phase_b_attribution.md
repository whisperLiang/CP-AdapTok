# HLA-Tok V3 B0 配对归因

正式 B0 在 20,000 步通过 K16320 连续门槛。H100 BF16 的 174 帧终点为 31.47002 / 0.10595 / 0.06576。
相对同批 A0 完整验证，PSNR 变化 -0.28654 dB，LPIPS-Alex 变化 +0.00556，DISTS 变化 -0.00129。

## 共享输入的三条重建路径

| B0 步数 | 证据设备 | 帧数 | P0 A0 稠密 | P1 B0 稠密旁路 | P2 B0 连续 Token |
|---:|---|---:|---|---|---|
| 2,000 | CPU FP32 | 9 | 31.78553 / 0.10614 / 0.06786 | 31.78553 / 0.10614 / 0.06786 | 22.99496 / 0.40105 / 0.28870 |
| 12,000 | CPU FP32 | 9 | 31.78553 / 0.10614 / 0.06786 | 20.82806 / 0.33814 / 0.20798 | 30.64842 / 0.12596 / 0.06605 |
| 20,000 | H100 BF16 | 174 | 31.75656 / 0.10040 / 0.06705 | 21.50744 / 0.32833 / 0.21478 | 31.47002 / 0.10595 / 0.06576 |

表中依次为 Global PSNR、LPIPS-Alex 和 DISTS。P1 与 P2 使用同一 B0 Encoder 前向、同一检查点和同一 RGB Decoder。

- 冻结接口期的 9 帧配对中，P0 与 P1 三项指标相同。P2 的 PSNR 相对 P1 变化 -8.791 dB，定位到经过 Tokenizer 与 Local Decoder 的整条瓶颈路径。
- H100 BF16 终点 P1−P0 的 PSNR 为 -10.249 dB，P2−P1 为 +9.963 dB。B0 的 RGB Decoder 在联合训练中适应了 Local Decoder 特征，该比较度量检查点上的实际旁路与正式连续路径。

## Rank 干预与潜在对齐 · B0 20,000 · H100 BF16 · 174 帧

| 干预 | 操作 | 接收 Token 数 | PSNR / LPIPS-Alex / DISTS |
|---|---|---:|---|
| regional_mean | token_content | 16320 | 16.17309 / 0.62203 / 0.48472 |
| repeat_rank0 | token_content | 16320 | 15.11966 / 0.59316 / 0.44780 |
| shuffle_values | token_content | 16320 | 28.85298 / 0.13282 / 0.08598 |
| perturb_rank_assignment | rank_assignment | 16320 | 29.94015 / 0.12078 / 0.07819 |
| delete_early16 | token_count | 14160 | 31.46759 / 0.10589 / 0.06575 |
| delete_middle16 | token_count | 14160 | 27.28515 / 0.15603 / 0.09244 |
| delete_late16 | token_count | 14160 | 23.84865 / 0.22580 / 0.12746 |

| 对齐路径 | 有效区 MSE | Padding MSE | 有效区余弦 |
|---|---:|---:|---:|
| b0_dense_vs_local | 0.206339 | 0.168044 | 0.80230 |
| phase_a_vs_b0_dense | 0.163567 | 0.141171 | 0.84540 |
| phase_a_vs_b0_local | 0.039548 | 0.020619 | 0.96624 |

相同的 2,160 个 Token 删除量下，早段近乎无损，中段下降 4.18 dB，晚段下降 7.62 dB。这些 Rank 干预对应各自的变化条件。原始记录含频带、内部分区边界、每通道方差、Token 统计及逐帧指标。 [同检查点原始表](continuous/seed1/B0/bottleneck/step_20000/full/metrics.csv)。

## 注册 Train 裁块的梯度审计

| 步数 | 设备 | Latent 权重 | 冻结 Encoder / RGB Decoder | Teacher 有梯度 | LPIPS-Alex / VGG / DISTS 的 RGB 梯度范数 |
|---:|---|---:|---|---|---|
| 2,000 | CPU BF16 | 0.1 | True | False | 0.13907 / 0.14226 / 0.13998 |
| 12,000 | CPU BF16 | 0.0 | False | False | 0.13891 / 0.20455 / 0.28447 |
| 20,000 | CPU BF16 | 0.0 | False | False | 0.05669 / 0.07172 / 0.07828 |
| 20,000 | H100 BF16 | 0.0 | False | False | 0.05652 / 0.07184 / 0.08249 |

训练版 `c844c25` 在原生 leg 的潜在权重为 0。Teacher 对象在 B0 阶段返回时释放。

ARD 为 NOT_APPLICABLE。正式准入以冻结协议的 H100 结果为准。
