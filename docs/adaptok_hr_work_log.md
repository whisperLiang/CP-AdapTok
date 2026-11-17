# AdapTok-HR 工作日志

执行规范：[ub-mode](/home/whisperliang/.codex/skills/ub-mode/SKILL.md)。每个可验证单元先运行行为测试，再实现、复测、审计。时间口径 Asia/Shanghai；实验精确时间另见 JSON。

| 操作 | 原因与证据 |
| --- | --- |
| 阅读完整任务、ub-mode、仓库状态和工作树 | 确认范围、保护用户数据；工作区干净，无适用 AGENTS.md。 |
| 读取 official / fixed2048 两分支报告与实现、原训练协议 | 官方 c935509 为封存推理对照；fixed2048 630e814 记录长序列负结果，不继承其模型。 |
| 核验 H100 PCIe 80GB、PyTorch 2.9.1+cu130、指标依赖与本地数据 | 单 GPU 空闲，原生高分辨率 Train 54 源/3827 帧；无需下载数据。 |
| 从 official c935509 建立 experiment/adaptok-hr | 独立 adaptok_hr 包；不修改 CP 编解码、子模块、历史权重或封存实验。 |
| 查询 crapper / dryer / mutator | 未发现命令或专用工具；后续使用 radon+coverage、AST 相似度及实际行为变异替代，明确标注。 |

不按 ub-mode 的一般建议推送 main：用户明确要求隔离实验分支并在有权限时推送该分支，优先遵守本任务范围。
