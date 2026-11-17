# Fine Token 恢复工作日志

实验分支：`experiment/adaptok-hr-fine-recovery`；基线：`cc87bd4`。
执行规则：ub-mode，先验证行为，再做最小实现；所有数值来自原始实验。

## 2026-11-18（Asia/Shanghai）

1. 读取用户任务与 ub-mode，检查 Git、数据、GPU、依赖和旧实验身份。原因：先确认可运行环境与历史保护边界。工作区干净，H100 80GB 可用，原生 1080p 数据及 LPIPS/DISTS 权重已存在。
2. 从指定提交创建独立分支。原因：保留原实验、Checkpoint、CP 和官方 AdapTok 不变；新产物只写入 fine_recovery 专用目录。
3. 阅读 Encoder/Tokenizer/VQ/Decoder、训练、Checkpoint、真实码流与评估代码。确认当前 VQ 对全部候选计算损失，辅助预算不推进历史。
4. 开始冻结旧 1000 步模型的 E0 诊断；只读取 Train/Validation。原因：先区分连续表征、离散量化和 Decoder 敏感性，再决定改进。

工具说明：环境没有名为 crapper/dryer/mutator 的工具；将使用已有 Python 覆盖率、radon、重复代码审计和隔离目录中的行为变异检查记录对应证据。不会把缺失工具记为已运行。
