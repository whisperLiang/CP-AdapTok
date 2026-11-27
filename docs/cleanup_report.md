# CP-AdapTok 清理验收

PHT-Tok 是唯一持续研发主线。原研究准入为 FAIL，原总体研究裁决为 PARTIAL_PASS。本次验收使用原始 PHT 提交 `54674cc`、D1 提交 `ffbb717` 和官方 AdapTok 分块实验提交 `c935509`。

四种方案在单卡 H100 上使用相同的 DAVIS2017 car-roundabout 前 17 个 Validation 帧。官方 LD、HT-S、PHT-Tok 与 D1 的实际码流及重建摘要逐值相等。AdapTok 的官方权重状态和两种分块的重建摘要一致。PHT-Tok 的 P1/P2 四前缀、独立参考、UVG 连续 16 个 P-Chunk、参考丢失后的 I 帧恢复和 Verizon 最小回放均已核对。D1 验证了 5/10/20/40 Mbps 的真实前缀及 Base 因果闭环。

未绑定 CPU 时，LD 解码 P95 有约 7% 的差异。用相同的 H100 本地 NUMA 核心、相同预热和计时代码，按 ABBA 顺序做八对独立测量。解码 P95 的中位数从 2.590 ms 到 2.563 ms，峰值显存相同。PHT-Tok 及 D1 的实际码流、质量和运行记录见 [H100 对照](../results/cleanup/h100_regression.json)。

权重 SHA256、保护数据划分、物理删除范围与不确定资产见 [精确清理计划](../cleanup_plan.json)。Test 仅核对清单和文件元数据。Test 像素读取计数为 0。工作树、分支与磁盘回收将在通过独立目录验收后逐项记录。
