# 复现实验

使用指定基础提交 `c935509ff6bd00390f2421d39ce8501774b11d4f` 的 HLA-Tok worktree。运行环境需要与数据清单绑定的原始 Train/Validation 路径、已核验的 AdapTok 权重、Joint V1 只读 worktree 和 H100。原版 AdapTok 使用官方基线虚拟环境，HLA-Tok 和 Joint V1 重放使用 PyTorch 2.9.1 CUDA 13 环境。

```bash
python -m pytest -q -m 'not gpu and not integration' cp_adaptok/tests experiments/original_adaptok_1080p/tests tests/hla_tok
python -m pytest -q -m gpu tests/hla_tok --junitxml=results/hla_tok/phase0/cuda_tests.xml
python -m ruff check hla_tok scripts/hla_tok tests/hla_tok
python scripts/hla_tok/mutation_check.py

python scripts/hla_tok/run_phase0.py audit --output results/hla_tok/reproduced_audit
python scripts/hla_tok/run_phase0.py benchmark --output results/hla_tok/reproduced_t0
python scripts/hla_tok/run_phase0.py pipeline --output results/hla_tok/reproduced_pipeline
```

用环境中的真实路径设置 `OFFICIAL_SOURCE`、`OFFICIAL_CHECKPOINT`、`JOINT_ROOT`、`OFFICIAL_PYTHON`。以下产物路径应为空目录或尚未存在。

```bash
"$OFFICIAL_PYTHON" scripts/hla_tok/run_phase0.py baselines \
  --official-source "$OFFICIAL_SOURCE" --official-checkpoint "$OFFICIAL_CHECKPOINT" \
  --joint-root "$JOINT_ROOT" --output results/hla_tok/reproduced_baselines

python scripts/hla_tok/replay_joint_validation.py \
  --joint-root "$JOINT_ROOT" --output results/hla_tok/reproduced_joint

for mode in cache compile; do
  "$OFFICIAL_PYTHON" scripts/hla_tok/diagnose_official.py "$mode" \
    --official-source "$OFFICIAL_SOURCE" --official-checkpoint "$OFFICIAL_CHECKPOINT" \
    --output "results/hla_tok/reproduced_official_$mode"
done
```

在 Phase 0 验收文件标为 PASS 后，以下命令运行固定预算诊断。使用相同配置、清单和 Git 提交从指定检查点续跑。训练和验证通过同一个只允许 Train/Validation 的数据入口统计像素读取次数。

```bash
python scripts/hla_tok/run_phase1.py --train --stage continuous \
  --output results/hla_tok/phase1/continuous --checkpoint-dir checkpoints/hla_tok/continuous

python scripts/hla_tok/run_phase1.py --train --stage fixed \
  --initialize-from checkpoints/hla_tok/continuous/step_00500.pt \
  --output results/hla_tok/phase1/fixed --checkpoint-dir checkpoints/hla_tok/fixed

python scripts/hla_tok/run_phase1.py --resume checkpoints/hla_tok/fixed/step_01500.pt \
  --output results/hla_tok/phase1/fixed --checkpoint-dir checkpoints/hla_tok/fixed

python scripts/hla_tok/run_phase1.py --evaluate checkpoints/hla_tok/fixed/step_04500.pt \
  --output results/hla_tok/reproduced_fixed
```

归档每次运行的 CSV、GPU 环境、读取计数及检查点哈希。产物所用正式路径见 [最终报告](../../results/hla_tok/final_report.md)。
