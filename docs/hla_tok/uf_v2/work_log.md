# UF-V2 工作日志

1. 从本地 A/B worktree 核验方案 B `3efe25e`、方案 A `a483a21` 和共同祖先 `502eaf6`。创建独立 V2 checkout，只读检查两份完整报告、机器表、源码、协议及 checkpoint。比对 Original 已封装 F13W 窗口账本与 174 帧实际字节。
2. 在同一 H100 和官方 FP16 扩展上运行 LD、Anchor、D0 性能采样。分别保存未插桩和插桩 wall time、Python/C++ 调用和 Kineto 可观察的传输。历史基线和正式吞吐范围来自各自完整 Validation 日志。
3. 在冻结的 174 帧上运行最低官方 QP 和 Base-Standard 固定 QP。完整解码保留探索性超限样本，主曲线按物理上限拒收。
4. 先用分组边缘、符号逃逸和真实前缀测试锁定 E0 wire，再把逐段 rANS 和分组换成可批处理的 CPU 路径。用独立收端逐字节、逐像素和基础 latent 对照历史运行时。
5. 保留官方 I 的头前重建 feature，P 的原生 reference feature 和当前帧 context 按值快照。两条 D1 分支共享原生重建头，零增强直接走 Base。记录首轮串行批处理的 40 Mbps 延迟失效，随后测量并行段打包和原生增量解析。
