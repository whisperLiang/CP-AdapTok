# AIP-VTok Joint V1

在 experiment/aip-vtok-sparse-decoder 的 c5cd06dfc8cb6afd19c6da668d7664eb3f32c2ea 上建立独立实验分支。模型原生处理 1920×1080 I/P 帧。对照与数据清单沿用封存资产。完整模块和梯度边界见 [架构契约](../results/aip_vtok/joint_v1/architecture_contract.md)。

## 复现

检出实验分支并初始化固定提交的 DCVC 子模块。使用现有 CUDA 环境和预训练 LPIPS-Alex / DISTS 权重，按原有目录部署冻结资产。程序逐项验证 [父实验 provenance](../results/aip_vtok/sparse_decoder/provenance.json) 中的冻结文件 SHA256、指标权重与骨干张量。数据清单使用已经声明的绝对像素路径。校验清单与帧哈希交集只读取元数据，帧像素逐项按清单校验。结果、权重和码流写入独立的 Joint V1 路径。

```bash
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --train
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --resume
python scripts/aip_vtok/evaluate_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --split validation
python scripts/aip_vtok/benchmark_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1
python scripts/aip_vtok/run_joint_v1.py --config configs/aip_vtok/joint_v1.yaml --seed 1 --pipeline
```

Pipeline 依次完成 seed 1 的训练、选择和 Validation，再以同样的配置执行 seed 2。一次正式 Test 绑定 seed 1 的冻结 checkpoint、模型指纹、扫描表、配置和评估代码哈希。随后执行不含指标网络计时的硬件 benchmark。SIGTERM 和 SIGINT 在安全更新边界保存 checkpoint。SIGUSR1 在下一安全边界保存 checkpoint 并继续训练。非正常退出后，resume 从最新已校验的 checkpoint 恢复数据游标、预算序列、优化器、调度器及随机状态，用 checkpoint 的日志哈希恢复提交边界。

## 固定实验选择

A/B 使用四帧窗口，C 使用八帧窗口。Train 原视频窗口按种子和 global step 确定。窗口按真实 GOP 相位回退到最近 I 帧做无梯度 burn-in。所有阶段的 TX/RX 各自从整数参考前缀重建并比对状态。显示损失按去重的抽样预算等权平均。参考损失比较固定前缀的投影特征与冻结 aligned DCVC 全潜在路径所生成的对应目标。Rate loss 按实际必须传输的 Token 集合累计代理 bit，再除以原始像素数。

AdamW 使用配置中的分组学习率、1,000-step warmup 和余弦衰减至峰值的 10%。从 step 24,001 起冻结并导出扫描表。正式模型选择仅在阶段 C 的预定 Validation 检查点内进行，依八档 CLOSED 的平均 MSE + 0.1 LPIPS + 0.1 DISTS + 0.02 实际 bpp 排名。并列时使用较早一步。seed 2 独立训练和选择。

## 码流与计时

帧头包含版本、绝对帧号、I/P、模型与扫描 SHA256。每个包记录起始 Token、包内数量、payload 字节数和 CRC32，payload 以一个 32-bit rANS 状态开头。分包在八档预算边界对齐，序列化使用小端序。参考帧和展示都使用同一份码流。较低 I 帧显示预算仍计入 I2048 参考前缀。FIXED 是给定历史条件下的诊断，端到端结论使用 CLOSED。

延迟基于逐段 CUDA 同步后测得的 elapsed time，包含 CPU rANS 和 TX/RX 参考计算。Entropy payload 计入独立 rANS 状态，header 字节另列。旧对照记录 raw 或估计码率。原始 PNG、整数 ID 和字节流由运行阶段写入本地，Git发布统计、哈希及可复现脚本。

[实验报告](../results/aip_vtok/joint_v1/final_report.md)
