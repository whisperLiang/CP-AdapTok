# 训练与恢复

正式 Stage 0–4 需要官方 LD/DMCI 权重。latent distillation 还需要真实 native encoder/decoder CUDA 扩展；preflight 会列出所有缺失条件并退出 2。两个官方 OneDrive 分享的匿名 API 均返回 HTTP 401，本机也没有 nvcc；未执行原生码流复现或正式蒸馏。

`NativeLatentTeacher` 使用独立的 native encoder/decoder 真正生成、解析并解码上游码流，再要求官方 PyTorch quantized prior 的重建与 native 像素输出满足显式容差。Proxy 不导出内部 y_hat，因此此路径是“官方量化 prior 表示经真实码流解码重建核对”，不是声称已从 Proxy 提取内部 latent。该数值核对尚未在本机执行；失败会停止训练，不能退回原始 encoder latent。teacher 只进入 loss，不进入学生 Decoder、reference 或码流。

| Stage | 实现 | 本机执行状态 |
|---|---|---|
| 0 | 独立 Proxy 实际码流 + 官方 SPS/IP 元数据、PSNR/感知指标/耗时/字节统计 | 缺官方权重/扩展 |
| 1 | 冻结官方骨干；训练 queries、VQ、统一 decoder；每帧抽 2–4 档 log-spaced 前缀，两端加权 | 100 步结构训练，非正式模型 |
| 2 | reconstructed base-only 闭环、4→8→16→32 帧课程、truncated BPTT | 短 BPTT 行为测试及 checkpoint 恢复；长训练未执行 |
| 3 | learned block probability tables + CausalScorer；实际解码失真 labels 和 prediction errors | 100 步结构训练；随机骨干不支持有效质量结论 |
| 4 | 按 YAML 模块名选择性解冻 LD/I Encoder/Decoder/ReconHead | 实现；正式微调未执行 |
| 5 | SDPA query attention、精确缓存、逐显示点重新重建的独立测量 | 已做缓存精度与实际字节流性能检查；compile/CUDA Graph/fused entropy 尚未验证 |

默认 crop 256，可设 512；smoke 明确改成 64 crop / 2 帧，关闭不可用的 native teacher 和感知训练 loss。结构模型 N_max=2048、dim=192、heads=6、VQ=8192×16 不缩小。每 100 步在独立 UVG 片段的完整 1080p 上验证；测试使用 BasketballDrive，训练使用 MCL-JCV01。划分检查拒绝跨 split 重复真实路径。仅使用短片不能证明数据集级泛化。

loss 包含 RGB MSE、可选 LPIPS/DISTS、latent 蒸馏、VQ、熵模型估计率以及相邻采样前缀的软单调惩罚。估计率与真实字节码率分别记录。V1 的硬 VQ indices 估计率对概率表训练有效，对 argmin 没有 encoder 梯度；正式 Stage 4 的 bottleneck 码率微调还需校准可微 rate surrogate/交替优化，这是明确的研究未完成项。

bf16 autocast、梯度累积、梯度裁剪、seed、独立验证、OOM memory summary 均已接入。参考 feature 在 BPTT chunk 边界 detach；增强预算采样不决定 reference。原生 teacher 在 autocast 之外运行，以免隐式 bf16 干扰其与 CUDA Proxy 的核对。

checkpoint 原子替换并保存 model、optimizer、配置、已完成 step、采样 generator、CPU/CUDA/Python RNG 和实验元数据。`--checkpoint` 不存在时直接失败，不能忽略显式路径；`--resume` 要求同一 stage。切阶段用 warm start，不复用上一阶段的 optimizer。短 BPTT 单测已验证连续第二次更新与保存恢复后更新逐位一致。

长期训练前需提供足够长的独立视频、检查 native teacher 容差、码本利用率、每档实际 RD 和更多 token 质量下降比例。不能只看训练 forward 的失真或最高预算。当前两个 100 步模型状态均为 `untrained_backbone_smoke`，正式训练验收尚未完成。
