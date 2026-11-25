# HLA-Tok-C 复现

从 `experiment/hla-tok-continuous` 工作区根目录执行。需要单卡 H100、支持 CUDA BF16 的 PyTorch 和原 V3 评估依赖。V3 资产目录应指向已冻结的 `b3b3366` 工作区，包含 C0 和 C1 终点及原始评价记录。配置在本分支首个提交固定。已有测量目录可只读审阅，复跑使用新的输出目录。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HLAC_V3=/path/to/hla-tok-v3/CP-AdapTok
HLAC_RUN=/path/to/new-output
HLAC_PYTHON=/path/to/venv/bin/python

"$HLAC_PYTHON" -m scripts.hla_tok.continuous.assets before --v3 "$HLAC_V3" --output "$HLAC_RUN"
for stage in phase0 native ordering dynamic wire-checks wire; do
    "$HLAC_PYTHON" -m scripts.hla_tok.continuous.evaluate --v3 "$HLAC_V3" --output "$HLAC_RUN" "$stage" || break
done
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.benchmark --v3 "$HLAC_V3" --output "$HLAC_RUN"
```

Phase 0 使用原 V3 CLI 的完整评价路径。Phase 1 在原生预算复现通过后运行，先做 O0、前缀和因果，再冻结 Train 排序，最后重放六条动态序列。Phase 2 首先通过真实字节验证，之后执行八档率失真、等 Payload 和候选带宽三组评价。序列化、接收处理和样例都依赖同一组 C0 权重和静态排序。运行器仅加载推理模型。

每个矩阵的完整来源表可以在同一输入、软件、Checkpoint、排序、精度、预算及 Wire dtype 的情况下恢复。物理测速按配置单独发布，再生成总表。原 V3 结果、数据清单、检查点与分支在另一工作区只读保留。

## 工程回归

基线测试需在本 worktree 按 Gitlink 初始化原 DCVC/AdapTok 子模块，并提供原 DCVC 权重的只读链接。使用原 V3 已安装的 Python 依赖运行。

```bash
export COVERAGE_FILE="$HLAC_RUN/.coverage"
mkdir -p "$HLAC_RUN/logs"
CUDA_VISIBLE_DEVICES="" "$HLAC_PYTHON" -m pytest -q cp_adaptok/tests tests -m 'not gpu and not integration' \
    --cov=hla_tok --cov=scripts.hla_tok --cov=cp_adaptok --cov-report= \
    --junitxml="$HLAC_RUN/logs/cpu_tests.xml"
"$HLAC_PYTHON" -m pytest -q cp_adaptok/tests tests -m 'gpu and not integration' \
    --cov=hla_tok --cov=scripts.hla_tok --cov=cp_adaptok --cov-append --cov-report= \
    --junitxml="$HLAC_RUN/logs/cuda_tests.xml"
"$HLAC_PYTHON" -m pytest -q cp_adaptok/tests/test_state_fingerprint.py::test_state_digest_is_exactly_equal_on_cpu_and_cuda \
    --junitxml="$HLAC_RUN/logs/cross_device.xml"
"$HLAC_PYTHON" -m coverage json -o "$HLAC_RUN/logs/coverage.json"
"$HLAC_PYTHON" -m scripts.quality_audit --coverage "$HLAC_RUN/logs/coverage.json" \
    --output "$HLAC_RUN/logs/repository_complexity.json"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.engineering --coverage "$HLAC_RUN/logs/coverage.json" --output "$HLAC_RUN"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.mutations --output "$HLAC_RUN"
ruff check hla_tok scripts/hla_tok tests/hla_tok | tee "$HLAC_RUN/logs/ruff.log"
mypy --explicit-package-bases --follow-imports=skip --ignore-missing-imports --python-version 3.12 hla_tok scripts/hla_tok | tee "$HLAC_RUN/logs/mypy.log"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.engineering --attest --output "$HLAC_RUN"
```

## 原生样例和报告

```bash
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.samples --v3 "$HLAC_V3" --output "$HLAC_RUN"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.figures --output "$HLAC_RUN"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.assets after --v3 "$HLAC_V3" --output "$HLAC_RUN"
"$HLAC_PYTHON" -m scripts.hla_tok.continuous.report --output "$HLAC_RUN"
```

绘图使用 Matplotlib。环境变量 `NATURE_FIGURE_SCRIPTS` 可指向 PDF 和多面板 QA 工具目录。七张图读取正式数据表，完整日志、报告输入和帧级 CSV 保留在输出目录。物理码流定义见 [协议](protocol.md)，训练和评估范围见 [实现说明](implementation.md)。

## 产物对应

- [正式报告与裁决](../../../results/hla_tok/continuous/final_report.md)涵盖全部阶段及主要表格。
- `scripts/hla_tok/continuous/evaluate.py` 负责 Phase 0、1、2 数据矩阵，`benchmark.py` 负责 H100 定时。
- `hla_tok/streaming/continuous.py` 负责连续编解码，`continuous_wire.py` 负责浮点字节和分段接收。
- `fixed_budget*.csv` 与 `closed_budget*.csv` 对应 O0。`ordered_fixed_budget*.csv` 与 `ordered_closed_budget*.csv` 对应 O1/O2。
- `rank_*`、`prefix_audit.json`、`causality_audit.json`、`dynamic_*` 为排序、功能与时间证据。
- `wire_rate_distortion*.csv`、`equal_rate_comparison*.csv`、`bandwidth_constrained_quality*.csv` 为物理字节及画质观测。
- `latency_breakdown.csv`、`memory_benchmark.csv`、`throughput_benchmark.csv` 为单卡 H100 测量。
- `figures/` 保存七张图和 QA 记录，`samples/` 保存固定原生帧、裁块及 SHA-256。
