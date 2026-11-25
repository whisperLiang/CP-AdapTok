# 复现 HLA-Tok-UF-V2

从独立的 `experiment/hla-tok-uf-v2` checkout 运行。实验使用官方 Gitlink `cbdae87a5445114cdc7f48816da63ea80bdeac40` 和两个经校验的官方 checkpoint。保留对应历史 worktree，仅从中读取 A/B 证据、权重及可选 ABI 一致的编译对象。当前主机示例使用现有 Python 3.12 / PyTorch 2.9.1+cu130 / H100 路径。将路径改为实际安装位置即可在同 ABI 环境重跑。

```bash
export PROJECT=/home/whisperliang/CP-AdapTok
export UF_HISTORY=/home/whisperliang/.codex/worktrees/hla-tok-uf/CP-AdapTok
export CC_HISTORY=/home/whisperliang/.codex/worktrees/hla-tok-cc/CP-AdapTok
export PYTHON_BIN="$PROJECT/.venv-native/bin/python"
export CUDA_HOME="$PROJECT/.build/cuda-packages/nvidia/cu13"
export CUTLASS_HOME="$PROJECT/.build/cutlass"
export UF_OFFICIAL_WEIGHTS="$PROJECT/checkpoints"
export UF_NATIVE_BUILD_CACHE="$PROJECT/.build/native-dcvc/src/layers/extensions/inference/build"
export OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=.build/uf_v2/native:.build/uf_v2/records:.build/uf_v2/legacy:.
OUT=results/hla_tok/uf_v2
CKPT=checkpoints/hla_tok/uf_v2
ANCHOR1="$UF_HISTORY/checkpoints/hla_tok/uf/isolated/anchor-seed1-step2000.pt"
ANCHOR2="$UF_HISTORY/checkpoints/hla_tok/uf/isolated/anchor-seed2-step2000.pt"

git submodule update --init third_party/DCVC
bash scripts/hla_tok/uf_v2/build_native.sh
bash scripts/hla_tok/uf_v2/build_legacy_rans.sh
bash scripts/hla_tok/uf_v2/build_records.sh
"$PYTHON_BIN" -m pytest -q tests/hla_tok/uf tests/hla_tok/uf_v2
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.history verify \
  --uf "$UF_HISTORY" --cc "$CC_HISTORY" --weights "$UF_OFFICIAL_WEIGHTS" --output "$OUT/baseline"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.parity \
  --weights "$UF_OFFICIAL_WEIGHTS" --anchor "$ANCHOR1" --frames 34 --output "$OUT/evidence/replay-parity.json"
```

使用新的输出目录重跑测量，以保留已提交的证据。历史数据查询只用冻结 Train/Calibration/Validation 计划。CUDA jobs 串行执行。

```bash
REPLAY="$OUT/replay"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.profile \
  --weights "$UF_OFFICIAL_WEIGHTS" --anchor "$ANCHOR1" --trace-dir .build/uf_v2/replay-traces \
  --output "$REPLAY/profiling/legacy.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.trace_summary \
  .build/uf_v2/replay-traces "$REPLAY/profiling/cuda_activities.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.base_sweep \
  --weights "$UF_OFFICIAL_WEIGHTS" --output "$REPLAY/base-validation"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.calibrate \
  --weights "$UF_OFFICIAL_WEIGHTS" --output "$REPLAY/entropy"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.diagnostics \
  --weights "$UF_OFFICIAL_WEIGHTS" --anchor "$ANCHOR1" "$ANCHOR2" --output "$REPLAY/perceptual"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.evaluate --method anchor-fast \
  --weights "$UF_OFFICIAL_WEIGHTS" --checkpoint "$ANCHOR1" --repetitions 3 \
  --quality 0 0.5 1 1.395 2 3 3.594 4 5 10 20 40 --output "$REPLAY/anchor-fast"
```

下面的命令从头重跑 P1。也可读取当前 checkout 保留的 `$CKPT/d1-seed1-step2000.pt` 和后续节点。

```bash
RCK=".build/uf_v2/replay-checkpoints"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.train --weights "$UF_OFFICIAL_WEIGHTS" --steps 500 \
  --checkpoints "$RCK" --output "$REPLAY/training-seed1"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.train --weights "$UF_OFFICIAL_WEIGHTS" --steps 2000 \
  --resume "$RCK/d1-seed1-step500.pt" --checkpoints "$RCK" --output "$REPLAY/training-seed1"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.evaluate --method d1 --weights "$UF_OFFICIAL_WEIGHTS" \
  --checkpoint "$RCK/d1-seed1-step2000.pt" --entropy-table "$OUT/entropy/static_cdf.json" \
  --repetitions 3 --quality 0 0.5 1 1.395 2 3 3.594 4 5 10 20 40 --output "$REPLAY/d1-static"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.train_joint --weights "$UF_OFFICIAL_WEIGHTS" \
  --steps 4000 --resume "$RCK/d1-seed1-step2000.pt" --checkpoints "$RCK" --output "$REPLAY/joint-seed1"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.dynamic --weights "$UF_OFFICIAL_WEIGHTS" \
  --checkpoint "$RCK/d1-seed1-step2000.pt" --entropy-table "$OUT/entropy/static_cdf.json" \
  --low --output "$REPLAY/dynamic"
```

P2 节点每次最多增加 2,000 步。使用上个节点的 `d1-joint-seed1-step*.pt` 作为 `--resume` 目标并在每个节点用冻结 174 帧复测。固定学习率阶段无独立 scheduler。训练不修改历史 Anchor/CC 权重。

宽口径 CPU 回归读取根 checkout 的已验证官方权重。若新 worktree 没有默认名称的本地权重，可为两个 `.pth.tar` 文件创建指向只读权重目录的本地符号链接。若清理本轮临时文件，只处理 `.build/uf_v2` 中已无进程持有的构建副本，保留历史和实验 checkpoint。

正式阶段以累计 8,000 步及对应冻结 Validation 作为输入。续跑脚本跳过已经写出 checkpoint 和验证文件的节点。每个新增节点重新验证物理四档前缀。下列调用使用本 worktree 的训练产物和归档，使用独立的 `REPLAY` 子目录保存重新测量的结果。

```bash
export UFV2_RESULTS="$OUT"
export UFV2_CHECKPOINTS="$CKPT"
bash scripts/hla_tok/uf_v2/run_joint.sh 1
bash scripts/hla_tok/uf_v2/run_formal.sh 1
BEST1="$CKPT/d1-joint-seed1-step20000.pt"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.evaluate --method d1 \
  --weights "$UF_OFFICIAL_WEIGHTS" --checkpoint "$BEST1" --repetitions 3 \
  --quality 0 0.5 1 1.395 2 3 3.594 4 5 10 20 40 --output "$REPLAY/formal-e0"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.dynamic --weights "$UF_OFFICIAL_WEIGHTS" \
  --checkpoint "$BEST1" --low --output "$REPLAY/dynamic-e0"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.incremental_bench --weights "$UF_OFFICIAL_WEIGHTS" \
  --checkpoint "$BEST1" --output "$REPLAY/incremental-e0"
decision=$("$PYTHON_BIN" -m scripts.hla_tok.uf_v2.seed_admission "$OUT" \
  --runtime-summary "$REPLAY/formal-e0/summary.json" --closure-summary "$REPLAY/dynamic-e0/summary.json" \
  --incremental-summary "$REPLAY/incremental-e0/summary.json")
if [[ "$decision" == ADMITTED ]]; then
  bash scripts/hla_tok/uf_v2/run_joint.sh 2
  bash scripts/hla_tok/uf_v2/run_formal.sh 2
fi
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.samples --weights "$UF_OFFICIAL_WEIGHTS" \
  --checkpoint "$BEST1" --anchor "$ANCHOR1" --output "$REPLAY/native-e0"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.parity --weights "$UF_OFFICIAL_WEIGHTS" \
  --anchor "$ANCHOR1" --d1-checkpoint "$CKPT/d1-seed1-step2000.pt" --frames 34 --output "$REPLAY/paired.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.training_summary "$OUT"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.rate_analysis "$OUT"
```

记录级整数写入计时使用独立编译目录。常规 Full-max 计时使用不含插桩的构建。

```bash
UFV2_PROFILE_CANVAS=1 UFV2_RECORD_BUILD_DIR=.build/uf_v2/records-profile bash scripts/hla_tok/uf_v2/build_records.sh
PYTHONPATH=".build/uf_v2/records-profile:$PYTHONPATH" "$PYTHON_BIN" -m scripts.hla_tok.uf_v2.canvas_profile \
  --samples "$REPLAY/dynamic-e0/samples" --checkpoint "$BEST1" --output "$REPLAY/incremental-canvas.json"
```
