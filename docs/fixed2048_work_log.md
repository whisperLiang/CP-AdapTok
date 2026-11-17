# Fixed-2048 工作日志

遵循 ub-mode：先写失败测试，最小实现，再修正并验证。所有日期使用 Asia/Shanghai。

- 2026-11-17：读取用户完整任务和 ub-mode；检查原工作区无修改，保留 original-adaptok 分支。核实 B0/B1 文件存在、H100 PCIe 80GB 空闲、磁盘约 555GB 可用。
- 获取最新 origin/main（534696b），创建独立 managed worktree 和 experiment/fixed2048-reconstruction 分支。DCVC 使用固定 submodule commit；正式权重和 Checkpoint 仅以输入链接复用，实验输出在独立目录。
- 阅读生产 Codec、Tokenizer、VQ、Decoder、CausalState、Backbone Adapter、runtime、training、large_scale、metrics 与 manifest 数据结构；不改变生产文件。
- 已确认 continuous 是归一化 16D 投影回 192D，eval forward 拒绝此策略。实验显式调用 project_features/project_out。原训练每帧以 128 Token 的独立解码推进参考；实验将使用完整重建 Feature。
- crapper / dryer / mutator 命令和同名工具不可用；使用仓库已有 radon + coverage + AST 审计方法及局部 mutation 验证记录实际检查，不声称运行不存在的工具。
- TDD 第一轮：codec/checkpoint/diagnostics/train 模块缺失时产生预期 ImportError；实现后 11 项测试通过。增加实际优化器中断恢复测试，恢复后的下一更新与未中断运行逐参数完全一致。
- 实际 H100 单帧 1080p BF16 前后向通过（约 5.8 GiB），I/P Latent 实测分别为 [1,256,68,120] / [1,128,68,120]。正式 Backbone 与官方权重逐 Tensor 完全一致。
- 发现 R0 I-frame 原始 Encoder y 不等于官方 Decoder 的连续输入尺度：官方 forward_prior_4x 中 q_scale_y_enc / q_scale_y_dec 合积必须保留。bear 首帧原始直通 7.7949dB，保留连续缩放 34.5022dB。按任务保留原始 R0，新增单独尺度对照，不改生产 Adapter、不冒称容量上界。
- Phase 0：从 B1 S4 严格加载，Validation 的 blackswan / car-roundabout / dog 各 8 帧，运行 I-only、相同历史配对、各自 Full2048 闭环及历史 Original128 闭环；所有图像指标真实测量，未读取 Test 像素。
