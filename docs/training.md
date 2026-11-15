# 训练与恢复

## 前缀专项（2026-11-16）

最新研究问题是前缀内容利用，早期码本完全塌缩已修复。下文100步阶段表保留历史口径；新专项实测及失败结论见 [报告](prefix_utilization_report.md)。全局 Softmax 仍为默认，`prefix_progressive_decoder.yaml` 仅用于失败结构消融的复现。

`configs/prefix_nested.yaml` 对一次 Encoder/VQ 的同一序列监督 K_ref、N_max 和随机中间档。`prefix_training.weights` 为每档显式正权重，按实际采样预算数取平均；可用 `weight_schedule: [{step: ..., weights: {...}}]` 调度。MSE、latent 和感知失真均应用该权重；估计率另行记录。B2 的端点0.6、中间1.8权重用于平衡三档采样下五个预算的期望曝光，并非只重视最大预算。

新模式的 `relu(D_high - D_low.detach() + margin)` 只直接推动高预算分支；默认 margin=0。`one_sided: false` 仅用于旧约束消融。共享参数仍可能间接影响基础质量，因此绝对基础失真、验证集低预算质量和内容干预均不能省略。残差 L2 对齐与高预算 MSE 等价，本轮没有增加重复的增量辅助损失。

```bash
.venv-native/bin/python -m scripts.compare_prefix_training --steps 100
.venv-native/bin/python -m scripts.compare_prefix_decoders --steps 100
.venv-native/bin/python -m scripts.run_prefix_validation --engineering-steps 2
```

这些命令保留完整配置与 checkpoint，重复运行同配置可恢复已完成的更新；前两个命令拒绝混合不一致配置。最后一条运行有限的512/720/原生1080p、4/8/16帧训练链路检查，再评估独立原生1080p视频。正式大规模训练质量门槛未通过，这些更新不能标成正式收敛。

`migrate_global_aggregation` 是显式迁移入口，只允许聚合方式变更：逐项复用同形参数，枚举新 gate 参数，合成完整 state_dict 后 `strict=True` 加载，重建 optimizer。标准加载仍严格拒绝架构不符。训练专用 `latest.pt` 含 optimizer/RNG；train-only MLE 熵校准的 `calibrated.pt` 为推理产物，禁止 `--resume`。

官方 LD/DMCI 权重已下载并验证，见 [权重记录](pretrained_weights.md)。独立 `.venv-native` 已成功构建两个原生扩展：Python3.12.13、PyTorch2.9.1/cu130、nvcc13.0.88、CUTLASS4.4.1。复现使用 `bash scripts/setup_native.sh`；固定的上游 Git submodule 保持原样。

加载现有 CP checkpoint 时，其完整 state_dict 会覆盖骨干参数，因此旧 smoke checkpoint 仍代表随机骨干。新一轮预训练骨干实验应从官方 LD/DMCI 权重初始化，使用新的保存路径，再训练 CP 新增模块；下载资源不会改变已经保存的实验结果。

`NativeLatentTeacher` 使用独立的 native encoder/decoder 生成、解析并解码真实码流，再从原生 decoder 复制其实际使用的量化 `m_y_hat`。最小导出补丁只应用到 `.build` 副本；返回值 detached clone，修改副本不影响 native 状态。先前近似重算 prior 的方法在真实核对中失败（0.05914>0.02），已删除并保留失败证据。缺少导出接口的旧扩展会在 preflight 中拒绝。teacher 只进入 loss，不进入学生 Decoder、reference 或码流。

| Stage | 实现 | 本机执行状态 |
|---|---|---|
| 0 | 独立 Proxy 实际码流 + 官方 SPS/IP 元数据、PSNR/感知指标/耗时/字节统计 | 官方QP32、8帧1080p实际质量与无插桩性能已测 |
| 1 | 冻结官方骨干；训练 queries、VQ、统一 decoder；每帧抽 2–4 档 log-spaced 前缀，两端加权 | 官方骨干+native teacher+LPIPS完成100步，恢复至101步；码本塌缩、画质不合格 |
| 2 | reconstructed base-only 闭环、4→8→16→32 帧课程、truncated BPTT | 短 BPTT 行为测试及 checkpoint 恢复；长训练未执行 |
| 3 | learned block probability tables + CausalScorer；实际解码失真 labels 和 prediction errors | 100 步结构训练；随机骨干不支持有效质量结论 |
| 4 | 按 YAML 模块名选择性解冻 LD/I Encoder/Decoder/ReconHead | 实现；正式微调未执行 |
| 5 | SDPA query attention、精确缓存、逐显示点重新重建的独立测量 | 已做缓存精度与实际字节流性能检查；CP侧compile/CUDA Graph/fused entropy 尚未验证 |

默认 crop 256，可设 512；smoke 明确改成 64 crop / 2 帧，关闭不可用的 native teacher 和感知训练 loss。结构模型 N_max=2048、dim=192、heads=6、VQ=8192×16 不缩小。每 100 步在独立 UVG 片段的完整 1080p 上验证；测试使用 BasketballDrive，训练使用 MCL-JCV01。划分检查拒绝跨 split 重复真实路径。仅使用短片不能证明数据集级泛化。

`pretrained_s1.yaml` 声明64/16/32帧独立训练/验证/测试，使用 `scripts/prepare_pretrained_data.py` 准备。每次更新在训练区间内随机取连续序列，沿用checkpointed generator，记录源路径与每帧位置；100步实际访问47个起点。声明的训练区间必须容纳当前课程序列长度，超长或非法长度在消耗采样 RNG 前拒绝。

loss 包含 RGB MSE、可选 LPIPS/DISTS、latent 蒸馏、VQ、熵模型估计率以及相邻采样前缀的软单调惩罚。估计率与真实字节码率分别记录。Stage 3 用硬 indices 的交叉熵训练概率表；Stage 1/2/4 使用距离 soft assignment 对概率表的期望 bit cost，按各预算的像素数归一化，向 tokenizer/VQ 提供可微 rate 梯度。行为测试验证其将特征推向低成本码字，但真实 RD 校准仍需正式模型实验。

bf16 autocast、梯度累积、梯度裁剪、seed、独立验证、OOM memory summary 均已接入。参考 feature 在 BPTT chunk 边界 detach；增强预算采样不决定 reference。原生 teacher 在 autocast 之外运行，以免隐式 bf16 干扰其与 CUDA Proxy 的核对。

checkpoint 原子替换并保存 model、optimizer、配置、已完成 step、采样 generator、CPU/CUDA/Python RNG 和实验元数据。`--checkpoint` 不存在时直接失败；`--resume` 要求同一 stage。辅助 teacher/感知网络初始化结束后再恢复 RNG，避免构造过程消耗已恢复状态。切阶段用 warm start，不复用上一阶段的 optimizer。CPU短BPTT测试逐位一致；两个真实CUDA恢复进程的所有RNG相同，但bf16更新的模型最大差约9.4e-6，未承诺GPU逐位相同。

以上为早期100步实验：`checkpoints/pretrained_s1/step100.pt` 的全部773个官方骨干 state tensor保持完全不变，后期perplexity为1，独立8帧PSNR约14.56dB，五档画质近乎相同。后续已修复码本使用机制，但前缀信息利用仍未通过。旧 smoke 模型仍是 `untrained_backbone_smoke`；历史指标见 [报告](pretrained_native_report.md)。
