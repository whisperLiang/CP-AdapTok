# AIP-VTok Sparse Latent Decoder

冻结起点为 `606216e768c277aa59b4dde6d3bd84247a02f905`。本轮分支为 `experiment/aip-vtok-sparse-decoder`。实验在独立 worktree 运行。主队列为 174 帧原生 1080p Validation。 [最终报告](../results/aip_vtok/sparse_decoder/final_report.md) · [输入 SHA256](../results/aip_vtok/sparse_decoder/provenance.json)

## 复现

在本轮分支的 worktree 中执行。官方 DCVC 与 AdapTok 子模块使用原 Gitlink。原冻结配置依赖已有的 DCVC 权重、Alignment、Projector、Fix-A step 1000 和感知权重。取自冻结工作区的本地权重可按原相对路径只读访问。Train 和 Validation 像素目录沿用冻结 manifest 指向的路径。文件、指标权重和编解码骨干的张量哈希在入口校验。

```bash
git submodule update --init
pilot_python=/home/whisperliang/CP-AdapTok/.venv-native/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py phase0
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py phase1
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py train
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py evaluate
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py latency
"$pilot_python" -u scripts/aip_vtok/run_sparse_decoder.py report
```

逐阶段重放会验证已发布的对应证据。训练初始化、模型形状、优化器、步骤、窗口、损失、冻结资产及 checkpoint 均归档。三个模型各有 step 0、100、300、500 的权重。 [检查点](../checkpoints/aip_vtok/sparse_decoder/) · [冻结采样计划](../results/aip_vtok/sparse_decoder/pilot_plan.json) · [证据校验](../results/aip_vtok/sparse_decoder/pilot_verification.json)

## 实验中的协议选择

旧 W1 的 500 步日志显示，其四帧监督窗口之前可能已有 4、12、20 或 26 帧参考历史。E2/E3/E4 沿用原 W1 的实际监督帧、视频排列和裁剪种子，在该窗口的首帧建立 I/P/P/P 参考链。按实际 I/P 类型使用 W1 预算采样。E1 在本轮统一队列、预算和指标下重新评估，训练对照保留上述历史差异。 [W1 原计划](../results/aip_vtok/sparse_rfsq/training_plan.json) · [本轮计划](../results/aip_vtok/sparse_decoder/pilot_plan.json)

FIXED 采用旧 A1/C1 高斯恢复的 I2048/P510 参考历史。五种恢复器共用逐帧 A2/SP-1 展示码流。CLOSED 分别推进五条参考链。用于 W1 历史回放及接收端行为的原函数直接复用。新入口验证 `606216e` 冻结文件、已发布资产与本轮证据的 SHA256。封存 `cpat_test` 图像读取为 0。原生回归套件另含独立的 MCL-JCV 编码测试夹具。 [最终评估记录](../results/aip_vtok/sparse_decoder/pilot_evaluation.json)

## ub-mode 工作记录

| 顺序 | 操作与目的 | 结果 |
| --- | --- | --- |
| 1 | 核对冻结提交、H100、子模块和本地权重，建独立 worktree | 原分支留在冻结提交 |
| 2 | 先执行归因与逐帧清单测试，确认实现缺失时失败 | 预期红灯，最小实现后 2 项通过 |
| 3 | 首次 `preflight` 对照 W1 序列化历史 | 退出 1。PyTorch checkpoint 使用预算元组，JSON 保存预算列表。按 JSON 表示归一化后复核成功 |
| 4 | `phase0` 运行 O0–O4 与 K510 交叉对照 | 退出 0。174 帧、3654 行、训练步数 0 |
| 5 | 先加入 T1–T15 合约再编写三个恢复网络 | 缺少模型时红灯。首次联机试跑发现首个 Train holdout 不足 33 帧。夹具改用首个满足真实长度的 holdout，19 项通过 |
| 6 | `phase1` 执行原回归与新合约，用 JUnit 和输入 SHA256 放行 | 退出 0。333 项原测试及 19 项新测试 |
| 7 | 复核原 W1 门控重放，检查核心模型的复杂度、重复代码与变异 | W1 判定逐项相等。核心恢复函数最高 CRAP 3.0，9/9 定向变异被检出 |
| 8 | `train` 使用同一冻结四帧计划执行 E2/E3/E4 的 Pilot | 各 500 次更新，保存四个里程碑并评估已冻结的 Train holdout |
| 9 | `evaluate` 和 `latency` 测量同预算质量、原 S2 裕量和 GPU 耗时 | 退出 0。8,700 条逐帧测量与 50 个 H100 延迟组合。B2、W1 复评逐项匹配封存记录 |
| 10 | `report` 核验并生成停靠证据，检查 GPU 计算进程 | 退出 0。旧资产和 12 份 Pilot 权重通过校验，GPU 已结束本轮计算 |

本机以 Ruff、Radon、coverage.py、AST 相似性和隔离的源码变异执行质量检查。CRAP 和相似性覆盖核心接收端，外围阶段使用端到端运行记录。 [阶段命令及退出状态](../results/aip_vtok/sparse_decoder/audits/) · [单测](../results/aip_vtok/sparse_decoder/phase1_tests.json) · [ub-mode 质量记录](../results/aip_vtok/sparse_decoder/ub_quality.json)
