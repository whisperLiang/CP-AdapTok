# AIP-VTok 操作日志

采用 `/home/whisperliang/.codex/skills/ub-mode/SKILL.md`：先建立可检验行为，再实现、实测、定位根因。仅在当前 Gate 通过后推进。

- 2026-11-19（Asia/Shanghai）：审计当前干净分支 `experiment/adaptok-hr-perceptual-recovery`；H100 PCIe 80GB 无计算进程，未终止任何其他进程；`.venv-native` 为 PyTorch 2.9.1+cu130。
- 从 `experiment/fixed2048-reconstruction` 的 `630e814` 创建 `experiment/aip-vtok`，保留官方权重、数据清单、I 帧尺度诊断与所有旧生产代码和封存结果。
- 阅读官方 DMCI/DMC-LD、已有纯 PyTorch Adapter、原生 Proxy、数据与指标实现。I 连续直通必须保留 `q_scale_y_enc * q_scale_y_dec`；P Decoder 显式依赖历史 context。I 参考为 pixel-unshuffle 的 192 通道，P 参考为 Decoder 的 256 通道。
- 检查数学边界：256→128 的线性映射不可能对任意 256 维输入精确可逆。将原生直通和对齐直通分别报告，使用训练数据拟合 PCA 初始化，实测该信息损失是否满足质量门槛；不将原生通路的成绩算到 128 通道通路。
- 仅审计测试集元数据，不读取 cows/goat/parkour 测试图像。Gate 0 使用训练集拟合与独立验证集闭环。
- crapper/dryer/mutator 可执行工具与 MCP 均不可用；沿用仓库已采用的 coverage/Radon、结构重复扫描及隔离行为变异检查，记录实际执行结果。
- Gate 0 先运行失败测试（缺包导入），保存 `results/aip_vtok/tests/gate0_red.txt`；实现几何、尺度对齐、参考单提交和投影器后 CPU 测试通过。真实 H100 测试验证 I/P 原生潜在/参考形状、冻结且检查点化的 Decoder 对对齐层的非零梯度。
- 从训练集校准 I/P 分布；I 使用 PCA 低秩初始化，P 使用可逆通道归一化。独立运行原生实际码流、原生连续直通和对齐直通的验证闭环，分别保存逐帧 JSON，不以连续路径冒充实际码率。
- 在训练集自身投影参考闭环上训练轻量投影器；全程未读封存测试像素。400 步 checkpoint、独立四流 RNG、预取计划及优化器已保存，可恢复。
- Gate 0 已由 `python -m aip_vtok.evaluation.report --require-pass` 根据原始 JSON 判定通过。数值见自动生成报告；没有触发加宽重测条件。进入 Gate 1 的首个 200 步冒烟。
- 原 `cp_adaptok` / `adaptok_hr` 非 GPU 回归共 164 项通过；官方代码和历史封存文件相对基线无差异。
