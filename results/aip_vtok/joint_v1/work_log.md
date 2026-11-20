# Joint V1 工作日志

1. 核对本地及远端 sparse-decoder 基线，建立独立工作树与分支。固定 DCVC/AdapTok gitlinks，校验骨干、投影器、A2/SP-1、SLD-500、指标权重及原数据清单。保留原有文件与旧训练资产。
2. 先写架构契约和行为测试。在实现前确认 transform、tokenizer、scan 测试因缺少 Joint 模块而失败。
3. 实现帧型专属正交变换、A2 整数 STE、受空间覆盖约束的共享扫描和 SLD 迁移。旧 SLD-500 在基础组兼容输入的 I/P×两预算探针上逐元素重现。
4. 实现独立 rANS 包与发送、接收会话。检查八档严格字节前缀、独立包往返、坏包和错误状态拒收、固定参考预算与展示隔离。
5. 在 H100 上验证真实 RGB、DISTS 和 rate 梯度、BF16 优化器更新、64 帧闭环和恢复后的模型及 AdamW 逐元素一致。
6. 构建正式三阶段训练、系统作业入口、逐帧原生 Validation 和受单次释放保护的 Test、基线比较及实测图表发布。命令帮助和代码检查见 engineering_tests.json。
