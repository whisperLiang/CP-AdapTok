# 官方 DCVC-UF LD 和 HT-S Calibration

官方 Gitlink 为 `cbdae87a5445114cdc7f48816da63ea80bdeac40`。[官方命令与样本](calibration.json)保留两种模型在四段原生 1080p Calibration 视频上的 QP 扫描、实际 I/P 字节及逐片段结果。[公开 UVG](public_control.json)保留 YUV420 对照。[权重与源码证据](upstream_manifest.json)记录原始 SHA256。

| 评估配置 | 用途 |
| --- | --- |
| Official | 官方 skip 0.15，reset 128。LD 单帧，HT-S 八帧，原始脚本生成和解析真实流 |
| Historical matched | skip 0，reset 32。LD 32 帧 GOP，HT-S 33 帧 GOP，保留逐 QP 表 |
| PHT direct | 与 PHT 同源、HT-S 官方完整码流，并在动态对照用 S1 自适应 QP |

历史等 26 dB 处，HT-S 和 LD 分别用 0.47661 和 0.25523 Mbps。LPIPS 0.45 在历史对齐配置中为 0.44831 和 0.32391 Mbps，在官方配置中为 0.44900 和 0.31201 Mbps。这些结论只适用于相应原生 Calibration 内容与重叠的低码率段。

[历史 QP 扫描](qp_sweep.csv) · [率失真曲线](native_rd.csv) · [逐点复现审计](official_calibration.json) · [动态直接对照与 PHT](../pht_tok/network_results.json)
