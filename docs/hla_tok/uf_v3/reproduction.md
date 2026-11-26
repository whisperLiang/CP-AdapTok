# 复现 HLA-Tok-UF-V3

使用独立 V3 分支和 H100 上的 Python 3.12、PyTorch 2.9.1+cu130 及锁定 DCVC Gitlink。官方权重及 V2 两个 Seed checkpoint 都由审计校验。全部实验要求与历史相同的冻结数据缓存。新输出使用独立目录。

```bash
cd /home/whisperliang/.codex/worktrees/hla-tok-uf-v3/CP-AdapTok
export PROJECT=/home/whisperliang/CP-AdapTok
export UFV2_HISTORY=/home/whisperliang/.codex/worktrees/hla-tok-uf-v2/CP-AdapTok
export PYTHON_BIN="$PROJECT/.venv-native/bin/python"
export UF_WEIGHTS="$PROJECT/checkpoints"
export UFV3_OUT=results/hla_tok/uf_v3/replay
export UFV3_CHECKPOINTS=checkpoints/hla_tok/uf_v3/replay
export OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=.build/uf_v3/native:.build/uf_v3/records:.build/uf_v3/legacy:.
export V2_INIT="$UFV2_HISTORY/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"

git submodule update --init third_party/DCVC
mkdir -p .build/uf_v3/{native,records,legacy}
cp -L "$UFV2_HISTORY"/.build/uf_v2/native/inference_extensions_cuda*.so .build/uf_v3/native/
cp -L "$UFV2_HISTORY"/.build/uf_v2/records/hla_uf_v2_records*.so .build/uf_v3/records/
cp -L "$UFV2_HISTORY"/.build/uf_v2/legacy/hla_uf_rans*.so .build/uf_v3/legacy/
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.audit --v2 "$UFV2_HISTORY" --weights "$UF_WEIGHTS" \
  --output "$UFV3_OUT/audit"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.capacity --v2 "$UFV2_HISTORY" --weights "$UF_WEIGHTS" \
  --split calibration --limit 12 --output "$UFV3_OUT/diagnostics/calibration"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.decision "$UFV3_OUT"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.capacity --v2 "$UFV2_HISTORY" --weights "$UF_WEIGHTS" \
  --split validation --d0-timing --output "$UFV3_OUT/diagnostics/validation"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.reference "$UFV3_OUT"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.contracts --weights "$UF_WEIGHTS" --checkpoint "$V2_INIT" \
  --d0-equivalence --output "$UFV3_OUT/diagnostics"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.entropy --v2 "$UFV2_HISTORY" --weights "$UF_WEIGHTS" \
  --output "$UFV3_OUT/entropy"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.lite --weights "$UF_WEIGHTS" --output "$UFV3_OUT/base_lite"
bash scripts/hla_tok/uf_v3/run_pilot.sh
```

若需要重建对照扩展，在 V2 checkout 按 V2 复现文档运行三个 build 命令，再将固定版本复制到 V3 的三个构建目录。SHA-256 记入 V3 审计清单。

`run_pilot.sh` 先实际完成两步，恢复同一优化器至累计 500 步，再恢复至 2,000 步。每个节点以冻结 Calibration 实际码流验收。如果 P2 未通过且冻结骨干仍未经联合训练，可以执行一次预定义联合探针，每个 Seed 的上限仍遵循冻结配置。

```bash
bash scripts/hla_tok/uf_v3/run_joint.sh
FIT="$(cat "$UFV3_OUT/training/candidate.txt")"
bash scripts/hla_tok/uf_v3/run_evaluation.sh "$FIT"

"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.evaluate --weights "$UF_WEIGHTS" --checkpoint "$V2_INIT" \
  --split validation --quality 5 10 20 40 --static "$UFV3_OUT/entropy/static_cdf.json" --output "$UFV3_OUT/evaluation/A2"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.evaluate --weights "$UF_WEIGHTS" --checkpoint "$V2_INIT" \
  --split validation --demand --static "$UFV3_OUT/entropy/static_cdf.json" --output "$UFV3_OUT/evaluation/A2_on_demand"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.contracts --weights "$UF_WEIGHTS" --checkpoint "$V2_INIT" \
  --static "$UFV3_OUT/entropy/static_cdf.json" --output "$UFV3_OUT/evaluation/A2_contracts"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.capacity --weights "$UF_WEIGHTS" --fitted "$FIT" \
  --split calibration --limit 12 --static "$UFV3_OUT/entropy/static_cdf.json" --output "$UFV3_OUT/diagnostics/post_joint"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.precision --weights "$UF_WEIGHTS" --checkpoint "$V2_INIT" \
  --static "$UFV3_OUT/entropy/static_cdf.json" --output "$UFV3_OUT/diagnostics"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.base_hyperprior --weights "$UF_WEIGHTS" --output "$UFV3_OUT/base_lite"
bash scripts/hla_tok/uf_v3/run_runtime_trial.sh "$FIT"
bash scripts/hla_tok/uf_v3/run_runtime_acceptance.sh "$FIT"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.checkpoint_ledger "$UFV3_OUT" "$UFV3_CHECKPOINTS"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.finalize "$UFV3_OUT" --checkpoint "$FIT"

export FIGURE_TOOLS=/home/whisperliang/.codex/skills/nature-figure/scripts
PYTHONPATH="$FIGURE_TOOLS:$PYTHONPATH" "$PYTHON_BIN" -m scripts.hla_tok.uf_v3.figures \
  "$UFV3_OUT" --qa .build/uf_v3/figure-replay-qa
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.audit --v2 "$UFV2_HISTORY" --weights "$UF_WEIGHTS" \
  --output "$UFV3_OUT/audit" --verify-history
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.manifest "$UFV3_OUT" "$FIT"
```

拟合模型使用 BF16 自动混合精度和 FP32 率失真计算。实际码流使用官方 FP16 CUDA Proxy、FP32 残差处理和 FP16 显示模块。所有 CUDA 性能测量串行执行。Test 在正式准入表通过之前封存。


当前归档的训练后候选是 `r0-probe-seed1-step2001.pt`。可直接从同一本地分支对该文件再做原生验证。缺少选定组包扩展时，可用 `UFV2_RECORD_BUILD_DIR=.build/uf_v3/pack-reuse bash scripts/hla_tok/uf_v2/build_records.sh` 编译本分支版本。

```bash
export UFV3_OUT=results/hla_tok/uf_v3/replay-eval
export PYTHONPATH=".build/uf_v3/pack-reuse:$PYTHONPATH"
FIT=checkpoints/hla_tok/uf_v3/r0-probe-seed1-step2001.pt
CDF=results/hla_tok/uf_v3/entropy/static_cdf.json
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.evaluate --weights "$UF_WEIGHTS" --checkpoint "$FIT" --static "$CDF" \
  --split validation --quality 0 0.5 1 1.3947 1.395 2 3 3.594 4 5 10 20 40 --output "$UFV3_OUT/full_max"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v3.evaluate --weights "$UF_WEIGHTS" --checkpoint "$FIT" --static "$CDF" \
  --split validation --demand --output "$UFV3_OUT/on_demand"
```

CPU、CUDA 与变异回归使用同一源代码和已核验的本地扩展。

```bash
export UF_OFFICIAL_WEIGHTS="$UF_WEIGHTS"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest -q -m 'not gpu' cp_adaptok/tests tests/hla_tok
"$PYTHON_BIN" -m pytest -q tests/hla_tok/uf tests/hla_tok/uf_v2 tests/hla_tok/uf_v3
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m scripts.hla_tok.uf_v2.mutations --output "$UFV3_OUT/qa/mutations-cpu"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m scripts.hla_tok.uf_v3.qa --output "$UFV3_OUT/qa/mutations-v3"
"$PYTHON_BIN" -m scripts.hla_tok.uf_v2.mutations --gpu --output "$UFV3_OUT/qa/mutations-cuda"
```

当前归档的最终回归使用选定构建，CPU 471 项通过、1 项跳过、30 项 GPU 标记用例被筛除。CUDA 入口单独执行 84 项并全部通过。首次报告写入失败的短训练额外保存了 2 次 AdamW 更新，留在本机 `.build/uf_v3/failed-p1-report`。该失败属于归档记录，正常复现无需重现。
