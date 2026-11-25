# HLA-Tok-C 图表

图 1、2、4、5 和 7 的画质观测取自冻结的 174 帧原生 Validation。每个数据点对应同一个 C0 检查点及同一批 RGB 源帧。全局 PSNR 由有效 RGB MSE 得到。LPIPS-Alex 和 DISTS 逐帧平均。图 3 的 Rank 分数为 12 个 Train 来源的同帧删除失真增量平均。图 6 为 H100 串行实际测量的分段平均，按三次共 3000 次执行汇总。图中无跨 Seed 误差估计。

| 图 | 内容 | 实测来源 |
|---|---|---|
| [1](01_budget_distortion.pdf) | Token 预算与三种失真，FIXED、CLOSED 和三种排序 | [O0](../fixed_budget.csv)、[O1/O2](../ordered_fixed_budget.csv) |
| [2](02_rate_distortion.pdf) | 实际 Payload 与三种失真，C1 同口径及原版 16 帧参考 | [码流](../wire_rate_distortion.csv)、[等 Payload](../equal_rate_comparison.csv)、[原版参考](../original_adaptok_context.csv) |
| [3](03_rank_importance.pdf) | 训练集 Rank 删除重要性与 Validation 低档 PSNR 增益 | [Train 归因](../rank_importance_train.csv)、[排序选择](../ordering_decision.json) |
| [4](04_temporal_stability.pdf) | 三个物理片段的逐帧质量及闭环差值 | [排序 FIXED](../ordered_fixed_budget_per_frame.csv)、[排序 CLOSED](../ordered_closed_budget_per_frame.csv) |
| [5](05_dynamic_budget.pdf) | 主排序六条预算轨迹及实测逐帧 PSNR | [动态逐帧](../dynamic_budget_per_frame.csv) |
| [6](06_latency_breakdown.pdf) | 真实码流分段平均与串行总时延 | [延迟明细](../latency_breakdown.csv) |
| [7](07_bandwidth_quality.pdf) | 30 FPS、四个目标带宽下的物理解码质量 | [带宽观测](../bandwidth_constrained_quality.csv) |

图 2 的原版参考来自 16 帧输入块，保留为归档上下文。图 4 和图 5 的横坐标遵循独立片段边界。图 6 的黑色短线是真实 wall-clock 总延迟，彩色堆叠为各段测量值。

所有最终 PDF 已逐图检查面板对齐、最小字号和文字碰撞，最小实测字号为 6 pt。保留 PDF、SVG 和 300 dpi PNG。 [最终图件审计](qa.json)
