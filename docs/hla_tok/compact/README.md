# HLA-Tok-CP 复现

从 `experiment/hla-tok-compact` 根目录运行。正式测量使用单卡 H100、Python 3.12、PyTorch 2.9.1+cu130。需要冻结的 V3 Seed 1 C0/C1 checkpoints、历史连续分支工作区和原版 AdapTok safetensors。输入数据由仓库内已冻结的 manifest 解析并逐帧校验。Test 仅核对清单身份。子模块使用 Gitlink 指定的提交。

```bash
git submodule update --init --recursive
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=4
COMPACT_PYTHON=/path/to/python
COMPACT_V3=/path/to/hla-tok-v3/CP-AdapTok
COMPACT_CONTINUOUS=/path/to/hla-tok-continuous/CP-AdapTok
COMPACT_OFFICIAL=/path/to/pinned/AdapTok/checkpoint
COMPACT_RUN=/path/to/new/run
COMPACT_DEPS=/path/to/local/deps
uv pip install --python "$COMPACT_PYTHON" --target "$COMPACT_DEPS" \
  lz4==4.4.5 zstandard==0.25.0 timm==1.0.15 einops==0.8.2 \
  transformers==4.50.0 huggingface-hub==0.30.2 safetensors==0.5.3 easydict==1.13 \
  tensorboard==2.20.0 pandas==2.3.3 decord==0.6.0 pulp==3.3.0
export PYTHONPATH="$COMPACT_DEPS"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.engineering before --output "$COMPACT_RUN" \
  --v3 "$COMPACT_V3" --continuous "$COMPACT_CONTINUOUS"
"$COMPACT_PYTHON" -m scripts.hla_tok.continuous.evaluate phase0 --v3 "$COMPACT_V3" --output "$COMPACT_RUN"
for stage in audit projection lossless evaluate checks decision train; do
  "$COMPACT_PYTHON" -m "scripts.hla_tok.compact.$stage" --v3 "$COMPACT_V3" --output "$COMPACT_RUN" || exit
done
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.checks --trained --v3 "$COMPACT_V3" --output "$COMPACT_RUN"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.references frame --v3 "$COMPACT_V3" --output "$COMPACT_RUN"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.references original --v3 "$COMPACT_V3" --output "$COMPACT_RUN" \
  --official-checkpoint "$COMPACT_OFFICIAL"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.benchmark --v3 "$COMPACT_V3" --output "$COMPACT_RUN"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.sample --v3 "$COMPACT_V3" --output "$COMPACT_RUN"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.engineering after --output "$COMPACT_RUN" \
  --v3 "$COMPACT_V3" --continuous "$COMPACT_CONTINUOUS"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.report --tables-only --output "$COMPACT_RUN"
NATURE_FIGURE_SCRIPTS=/path/to/nature-figure/scripts "$COMPACT_PYTHON" -m scripts.hla_tok.compact.figures --output "$COMPACT_RUN"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.report --output "$COMPACT_RUN"
```

运行评估器前先确认复现达到原始容差。Train 固定清单见 `configs/hla_tok/compact/train_frames.json`。阶段输出会验证清单、配置、模型及静态 Profile 的 SHA-256。完整评估按视频断点续跑，测速按独立配置续跑。修改测量实现时使用新输出目录。

## 工程验证

用仓库原有 DCVC 权重与子模块执行。默认关闭非相关上游集成测试。原有容差沿用冻结项目配置。

```bash
mkdir -p "$COMPACT_RUN/logs"
"$COMPACT_PYTHON" scripts/prepare_compile_env.py
export TRITON_LIBCUDA_PATH="$PWD/.build/driver"
export COVERAGE_FILE="$COMPACT_RUN/.coverage"
CUDA_VISIBLE_DEVICES='' "$COMPACT_PYTHON" -m pytest -q cp_adaptok/tests tests -m 'not gpu and not integration' \
  --cov=hla_tok --cov=scripts.hla_tok --cov=cp_adaptok --cov-report= --junitxml="$COMPACT_RUN/logs/cpu_tests.xml"
"$COMPACT_PYTHON" -m pytest -q cp_adaptok/tests tests -m 'gpu and not integration' \
  --cov=hla_tok --cov=scripts.hla_tok --cov=cp_adaptok --cov-append --cov-report= --junitxml="$COMPACT_RUN/logs/cuda_tests.xml"
"$COMPACT_PYTHON" -m pytest -q cp_adaptok/tests/test_state_fingerprint.py::test_state_digest_is_exactly_equal_on_cpu_and_cuda \
  --junitxml="$COMPACT_RUN/logs/cross_device.xml"
"$COMPACT_PYTHON" -m scripts.hla_tok.compact.mutations --output "$COMPACT_RUN"
mkdir -p target/coverage/python
"$COMPACT_PYTHON" -m coverage lcov -o target/coverage/python/lcov.info
crapper --use-existing-coverage --threshold 10 hla_tok/compression hla_tok/streaming/compact.py hla_tok/streaming/compact_wire.py
dryer --threshold 0.8 hla_tok/compression hla_tok/streaming/compact.py hla_tok/streaming/compact_wire.py
ruff check hla_tok scripts/hla_tok tests/hla_tok
mypy --explicit-package-bases --follow-imports=skip --ignore-missing-imports --python-version 3.12 hla_tok scripts/hla_tok
```

[协议](protocol.md) · [实现说明](implementation.md) · [工作记录](work_log.md) · [最终报告](../../../results/hla_tok/compact/final_report.md)
