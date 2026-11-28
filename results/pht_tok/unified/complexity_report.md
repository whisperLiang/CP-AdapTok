# 代码质量验收

Ruff 与 mypy 通过。CPU 76 项、PHT CUDA 16 项、官方 AdapTok CUDA 1 项通过。定向变异分别检出 CPU 10/10、CUDA 11/11。Mutator 对 wire 版本、索引和损坏状态的定向算子子集检出 10/10。Dryer 相似度阈值 0.8，候选为 0。 [测试与变异数据](quality/summary.json) [Dryer](quality/dryer.txt)

| 指标 | 数量 |
|---|---:|
| 第一方生产与评估 Python 物理行数 | 5873 |
| 共享 PHT 包物理行数 | 1288 |
| LD 独占后端 Python / CUDA patch 新增行 | 13 / 135 |
| HT-S 独占后端 Python / CUDA patch 新增行 | 48 / 222 |
| 公共包导出 | 2 |
| Crapper 分析函数 | 383 |
| CRAP 达到或超过 10 | 92 |
| 上一项中 PHT 包函数 | 5 |
| 圈复杂度达到或超过 10 | 5 |

Python 物理行数含注释与空行，排除测试、第三方、生成数据和历史报告。共享 PHT 口径为包内减去两个原生后端类及 HT-S 八帧装配。原始 P2 精度细化与公共 rANS 实现在同一模块。Crapper 只合并实际 CPU/CUDA pytest 覆盖，单独运行的 H100 实验与图表未计入该覆盖。CRAP 低于 10 的全仓目标未达，最高项主要是仍支持历史重放的评估入口。 [完整 Crapper](quality/crapper.txt)

| 最高项 | 圈复杂度 | 覆盖 | CRAP |
|---|---:|---:|---:|
| `experiments.pht_tok.study.fixed_point` | 21 | 0.0% | 462.0 |
| `experiments.pht_tok.network.simulate` | 20 | 0.0% | 420.0 |
| `experiments.pht_tok.closed.run_closed` | 18 | 0.0% | 342.0 |
| `experiments.pht_tok.study.equivalence` | 12 | 0.0% | 156.0 |
| `baselines.adaptok.cli.configuration` | 9 | 0.0% | 90.0 |
| `experiments.pht_tok.evaluate.progressive_rd` | 9 | 0.0% | 90.0 |
| `experiments.pht_tok.evaluate.push_prefixes` | 9 | 0.0% | 90.0 |
