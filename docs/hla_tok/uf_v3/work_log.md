# V3 工作记录

- 从本地 V2 提交创建独立 V3 worktree 和分支。初始化锁定的 DCVC Gitlink，并把 ABI 一致的 V2 原生扩展复制到 V3 构建目录。
- 逐行核对 V2 阶段日志。P1 中间节点保存重叠快照，按相同内容去重。用 AdamW 张量计数核对步数，用逐张量哈希核对实际更新。
- 审计单测先因模块缺失失败，补齐后通过。在 H100 用冻结 Train 计划重放 V2 三阶段和 CUDA 权重折叠。
- 实际回放 Calibration 和 Validation 的 C0–C3，写出逐帧质量、合法增强容量与配对差。第一次 Calibration 全窗诊断遇到 I 帧在 2 Mbps 超顶，加入明确的预算与下游失步状态。机制诊断使用四个冻结窗的前 12 帧。
- 在同一 Seed 2 量化与 Base 参数下跑 D0-Fast 的 174 帧三轮 Full-max。核对 Original AdapTok 的 1,044 个档位与源帧键、实际窗口字节及指标聚合。
- 从最终 V2 Seed 2 的 128 张 Calibration 帧重拟合 CDF。固定整数和组序，核对 E0/E1/E2 实际应用字节、D1 RGB 和 Full-max。按冻结规则选 E1。
