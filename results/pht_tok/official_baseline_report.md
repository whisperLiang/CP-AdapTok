# 官方基线复现

固定 Microsoft DCVC 提交 `cbdae87a5445114cdc7f48816da63ea80bdeac40` 后，使用上游 `test_video.py` 在四段原生 1080p DAVIS Calibration 视频，以及 129 帧 UVG ShakeNDry 1080p YUV420 上重跑 LD、HT-S 和 HT-L。每种模型采用匹配的 I/P QP 0、16、32、48、63，skip 阈值 0.15，初始 I 帧和 128 帧 reset。Calibration 的 60 份完整流与历史归档 SHA256 相同。另用 DCVC-RT 的原始测试入口和官方权重跑了同一 Calibration 的 20 个序列/QP 点。

| Calibration 原始控制 | QP 0 Mbps | QP 16 Mbps | QP 32 Mbps | QP 48 Mbps | QP 63 Mbps |
|---|---:|---:|---:|---:|---:|
| LD | 0.22687 | 0.64218 | 1.89178 | 5.34836 | 12.49135 |
| HT-S | 0.14422 | 0.46027 | 1.37314 | 3.85910 | 8.33888 |
| HT-L | 0.13773 | 0.43531 | 1.28333 | 3.60664 | 7.83800 |

## 历史低码率差距

历史 HT-S 在 pooled RGB PSNR 26 dB 处需 0.47661 Mbps，LD 需 0.25523 Mbps。这是四段历史 RGB Calibration 的等画质结果。历史 0.44831 和 0.32391 Mbps 的 LPIPS-Alex 0.45 对照使用 matched 配置。official 配置对应 0.44900 和 0.31201 Mbps。两套历史结果均保留。官方论文对其验证集的平均结论适用于论文协议。当前 RGB 短片段、低码率和感知指标属于单独的实测范围。 [官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Ultra-Fast_Neural_Video_Compression_CVPR_2026_paper.html)

审计覆盖连续真实输入、官方权重、八帧联合、初始 I 帧、DPB、QP 方向、skip/reset、尾帧、物理 SPS/IP、分母、颜色和插值。四个 32 帧短片段中每个实例含一个 I 帧、三个完整 P 块、一个七帧尾块和一帧填充。共享 P 载荷每块只累计一次。物理率分母使用有效帧。UVG 使用原生 120 FPS。DAVIS 使用清单名义 30 FPS，原始采集时钟缺少独立记录。官方入口的逐帧平均 PSNR 与研究对照的 pooled PSNR 分别保留。RGB 和 YUV420 结果按来源分别报告。等质量计算仅在各自有效覆盖区间内进行。

Calibration 与官方 UVG 重跑保留原始逐帧 PSNR 和真实文件字节。Calibration 启动时的官方 CPU MS-SSIM 探针耗时较长，批量官方重跑将该选项关闭。渐进同源对照使用 GPU MS-SSIM。官方默认速度入口的 HEVC-B 媒体在本机未配置，速度对照使用同 H100 上的 UVG 1080p 原生输入。

DCVC-RT 的构建副本为 CUDA 13/PyTorch 的 pybind11 增加了向量比较模板的类型约束。实际测试使用官方 RT 权重和 CUDA kernel。补丁、原始源码和二进制摘要均保留。 [官方 DCVC-RT](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/DCVC-family/DCVC-RT/README.md)

- [官方逐序列结果及物理账本](official_baseline_audit.json)
- [固定版本与资产 SHA256](upstream_manifest.json)

## 历史问题逐项审计

| 核查对象 | 结果或证据 |
|---|---|
| 1、3 连续八帧和重复帧 | Calibration 清单按源序号连续读取。每个完整 P 块由八个源帧组成。尾块单独标记。 |
| 2 HT-S 权重 | 发布文件校验大小、QuickXorHash 和 SHA256，加载采用严格匹配。 |
| 4 官方 CUDA | 真实流走原生 proxy 和 C++ rANS。四段 Calibration 与既有原生字节完全相同。 |
| 5、6 DPB 和初始 I | 官方入口先编码独立 I 帧，P 块维持历史。UVG 的第 16 个 P 块按 128 帧周期重置记忆。 |
| 7、8 I/P QP 和方向 | 相同 QP 调度下三种 UF 模型的字节与质量随 0→63 上升。历史混合 QP 曲线按各自 I/P 设置保留。 |
| 9–11 intra、reset、skip | official 采用首 I、reset 128、skip 0.15。历史 matched 配置另外保留，两个口径分别引用。 |
| 12–14 补帧、bpp 分母和共享载荷 | 按每块一次物理 NAL 的长度计算，七帧尾块记一帧填充，分母取有效帧。 |
| 15、16 色彩和 PSNR | DAVIS 用官方 RGB 分支，UVG 用原始 YUV420 分支。逐帧 PSNR 和 pooled RGB PSNR 的聚合定义均保留。 |
| 17 FPS | DAVIS 名义 30 FPS，UVG 120 FPS。Mbps 用各自有效时长复算。 |
| 18 插值 | 只在两个包络实际重叠区间计算，仍缺工作点时输出 INSUFFICIENT_OVERLAP。 |
