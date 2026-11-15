# Prefix information utilization 操作记录

全部时间使用 Asia/Shanghai；原始训练日志、配置、检查点及失败证据保留在 `results/prefix_utilization/`。

1. 读取用户任务全文、ub-mode、当前源码及既有优化报告。原因：以当前实现和实测资产为依据。
2. 核查工作区干净、HEAD=`b6dba05933f52dba919438d2af8c1b6e086d14eb`，无额外 AGENTS.md；检查 H100、磁盘、独立数据与 native 环境。原因：保护已有资产并确定可执行实验范围。
3. 建立独立实验目录，运行旧测试并记录四个离散训练 checkpoint 元数据和哈希。选用上轮最终 `auxiliary/latest.pt`，不使用 continuous diagnostic。原因：诊断必须代表实际离散码流模型。
4. 先运行失败测试，再实现分组统计、内容/位置分离、冻结历史的六类内容干预、全局注意力贡献分解与五预算真实 autograd。完成原生 1080p 四帧诊断，并补查时间更新的 capacity/learned_512 离散模型。原因：验证瓶颈而非假定提示词结论。
5. 原测试 125 通过；新增诊断/嵌套/梯度测试 18 通过。新增序列 bootstrap 测试先失败后通过（2 项）。原因：工程正确和统计口径必须先可验证。
6. B0/B1/B2/B3/scheduled 均从同一离散 checkpoint 开始，真实固定四帧、256 crop、100 更新。每个模型另用训练 Token 做加一平滑 MLE 熵校准，然后原生 1080p Beauty 真实 rANS 验证。记录实际各预算监督次数；等步数不声称 FLOPs 完全相等。
7. B1/B2/B3 固定片段最大/最小增益仅 0.010–0.015 dB，独立验证 0.047–0.057 dB；不足以建立有效渐进表示。梯度约随 1/K 稀释、P 后缀置换几乎不改变图像、共享前缀梯度近乎同向。由此进入 C 进行共享分组聚合的受控结构消融，原版仍为默认。
8. 检查增量 L2 残差对齐与普通高预算 MSE 的实际值/梯度等价性，不增加重复的 B4 损失。尝试本地 DAVIS ZIP 发现其为旧的不完整下载，保持原文件，改用已有 HTTP range/CRC 工具提取未训练的官方 val 序列。

ub-mode 指定的 crapper/dryer/mutator 未提供；使用仓库已有 radon、coverage、AST 重复检测及隔离行为变异工具，明确记录替代工具的边界。
