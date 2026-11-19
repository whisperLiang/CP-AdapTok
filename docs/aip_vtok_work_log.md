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
- Gate 1 先保存 FSQ/扫描/损失/冒烟放行的失败测试，再实现 64,000 符号双射、偶数 FSQ 半格偏移、确定性 Halton/Tier 扫描、三段量化后真残差与单次编码 API。
- 逐包画布与稠密重建精确比较暴露 GEMM 按批量尺寸切换归约的浮点尾差；以固定六项归约修复升维投影，不放宽为近似一致。ID=0 通过显式到达掩码参与重建。
- 真实 H100 完整原生帧集成测试验证 8192 Token 单次编码、510/2048/8192 P 帧增强不改变已提交参考、7-Token 与 128-Token 分块的潜在画布和最终 RGB 均逐元素完全相等、提交参考不调用 RGB Decoder。
- 冒烟参数在训练前固定：bear 训练视频首四帧、原生 1080p、200 步、三档配对损失、码率权重为零。放行需最后十步平均损失较前十步至少下降 20%，且 510→2048→8192 两阶各提升至少 0.5 dB，LPIPS/DISTS 同步改善。失败则不启动双种子长训。
- 200 步冒烟已完成并通过预先阈值。全部五预算曲线、每层符号熵、逐项损失梯度与模型梯度保存在 `results/aip_vtok/gate1_smoke/`；仍然只是训练集四帧，不冒充正式验证。
- 实测训练共享上下文路径：与原配对实现损失完全相等，最大参数梯度误差低于 1e-6。消除同一训练帧重复的历史网络执行和分包散射；推理增量 API 不变。另验证 BF16 激活训练梯度有限，标量 FSQ 保留 FP32，正式验证保持 FP32。
- 正式 Gate 1 协议在训练前固定为全原生 1080p、4 帧序列、每种子 1000 更新、seed=1/2、35 个训练视频，两个种子均从各自新初始化出发，不复用单视频过拟合权重。验证固定为 blackswan 50 + car-roundabout 64 + dog 60，共 174 帧；每档同时运行独立闭环与同状态配对，原封存测试集继续保持不打开。
