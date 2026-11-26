# HLA-Tok-HTS 复现

## 工作区与运行环境

本实验从 `5a415c98a57c4198f2e792974c14275217ad5952` 派生，使用独立的 `experiment/hla-tok-hts` worktree。官方 DCVC 源码提交、四份官方权重和研究门槛固定在 `configs/hla_tok/hts`。

```bash
export HTS_PROJECT=/home/whisperliang/CP-AdapTok
export HTS_HISTORY_V2=/home/whisperliang/.codex/worktrees/hla-tok-uf-v2/CP-AdapTok
export HTS_HISTORY_V3=/home/whisperliang/.codex/worktrees/hla-tok-uf-v3/CP-AdapTok
export HTS_WORKTREE=/home/whisperliang/.codex/worktrees/hla-tok-hts/CP-AdapTok
cd "$HTS_WORKTREE"
git submodule update --init third_party/DCVC third_party/AdapTok

export CUDA_VISIBLE_DEVICES=0
export OMP_WAIT_POLICY=PASSIVE
export PYTHON_BIN="$HTS_PROJECT/.venv-native/bin/python"
export ORIGINAL_PYTHON_BIN="$HTS_PROJECT/.venv-adaptok-official/bin/python"
export CUTLASS_HOME="$HTS_PROJECT/.build/cutlass"
export CUDA_HOME="$HTS_PROJECT/.build/cuda-packages/nvidia/cu13"
export HTS_WEIGHTS=checkpoints/hla_tok/hts/official
export HTS_RESULTS=results/hla_tok/hts
export PYTHONPATH=.build/hts/native/site:.
```

编码环境使用 Python 3.12.13、PyTorch 2.9.1+cu130、CUDA Toolkit 13.0.88、GCC 12.3.0 和 CUTLASS v4.4.1。Original AdapTok 使用历史 Python 3.10 环境和相同感知模型权重。CUDA 扩展从固定官方源码在 `.build/hts/native` 中构建。增量编译从已审阅的本机缓存重用源码相同的编译单元。

```bash
python -m scripts.hla_tok.hts.download_weights \
  --weights "$HTS_WEIGHTS" --manifest .build/hts/official_weights.json
HTS_NATIVE_CACHE="$HTS_PROJECT/.build/native-dcvc" bash scripts/hla_tok/hts/build_native.sh
```

在新的 H100 环境中可单独创建 Python 环境并安装锁定版本。先准备 CUDA Toolkit 13.0.88，然后将 `CUDA_HOME` 指向该工具包。将 `CUTLASS_HOME` 指向固定 CUTLASS checkout。

```bash
uv venv --python 3.12.13 .venv-native
export PYTHON_BIN="$PWD/.venv-native/bin/python"
uv pip install --python "$PYTHON_BIN" --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1+cu130 torchvision==0.24.1+cu130
uv pip install --python "$PYTHON_BIN" -r configs/hla_tok/hts/requirements-native.txt
git clone --branch v4.4.1 https://github.com/NVIDIA/cutlass.git .build/hts/cutlass
export CUTLASS_HOME="$PWD/.build/hts/cutlass"
git -C "$CUTLASS_HOME" checkout --detach 4370102f9dacab813282e1d67722fceb0b90a019
bash scripts/hla_tok/hts/build_native.sh
```

完整构建使用 `bash scripts/hla_tok/hts/build_native.sh`。下载器需要 NumPy 和 Requests。校验覆盖 OneDrive 提供的文件大小、QuickXorHash、ZIP CRC 及 SHA256。复现时与 `configs/hla_tok/hts/official_weights.json` 对照。 [DCVC-UF 发布说明](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/README.md#-usage)

## Phase 0

```bash
"$PYTHON_BIN" -m scripts.hla_tok.hts.native_contracts \
  --weights "$HTS_WEIGHTS" --output "$HTS_RESULTS/native_audit"
"$PYTHON_BIN" -u -m scripts.hla_tok.hts.audit_native \
  --weights "$HTS_WEIGHTS" --output "$HTS_RESULTS/native_audit" \
  --models hts ld htl --mixed
bash scripts/hla_tok/hts/phase0_controls.sh
"$PYTHON_BIN" -m scripts.hla_tok.hts.verify_ledger "$HTS_RESULTS/native_audit"
"$PYTHON_BIN" -m scripts.hla_tok.hts.native_comparison "$HTS_RESULTS/native_audit"
"$PYTHON_BIN" -m scripts.hla_tok.hts.phase0_gate "$HTS_RESULTS/native_audit"
```

每个点保留四段完整 SPS/IP 流。各片段通过临时文件提交，重新启动可从已有片段继续。审计与对照实验只读取锁定的 128 个 Calibration 帧。正式 Validation 的 174 帧与独立长序列以历史清单为准。

## 回归入口

```bash
ruff check hla_tok/hts scripts/hla_tok/hts tests/hla_tok/hts
mypy --explicit-package-bases --follow-imports=silent --ignore-missing-imports \
  hla_tok/hts scripts/hla_tok/hts
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m pytest -q -m 'not gpu' tests/hla_tok/hts
"$PYTHON_BIN" -m pytest -q -m gpu tests/hla_tok/hts
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m scripts.hla_tok.hts.mutations \
  --output "$HTS_RESULTS/qa/mutations-final"
"$PYTHON_BIN" -m scripts.hla_tok.hts.mutations --gpu \
  --output "$HTS_RESULTS/qa/mutations-gpu-verified"
```

## 图件与裁决

```bash
export FIGURE_TOOLS=/home/whisperliang/.codex/skills/nature-figure/scripts
PYTHONPATH="$FIGURE_TOOLS:$PYTHONPATH" "$PYTHON_BIN" -m scripts.hla_tok.hts.figures "$HTS_RESULTS" \
  --output "$HTS_RESULTS/figures" --qa "$HTS_RESULTS/qa/figures"
downloaded_logs="$HTS_RESULTS/logs"
"$PYTHON_BIN" -m scripts.hla_tok.hts.qa_summary "$HTS_RESULTS" "$downloaded_logs"
```

Phase 0 的实测裁决为 `FAIL` 时，可运行 `"$PYTHON_BIN" -m scripts.hla_tok.hts.finalize "$HTS_RESULTS"` 汇总结果。图件范围为已执行的原生与历史对照。
