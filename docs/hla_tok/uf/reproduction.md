# Reproduce HLA-Tok-UF

Run from the supplied experiment worktree, not the historical worktree. The tested runtime is Python 3.12.13, PyTorch 2.9.1+cu130, CUDA 13.0.88, GCC 12.3, the native `sm_90a` backend, and a single NVIDIA H100 PCIe. Use the official weights listed with digests in `configs/hla_tok/uf/upstream.json` and the archived frozen data manifests. `WEIGHTS` below points to the directory holding both native `.pth.tar` files. `CUDA_HOME` and `CUTLASS_HOME` point to the matching CUDA toolkit and audited CUTLASS checkout. Do not set `UF_NATIVE_BUILD_CACHE` for a different gitlink, CUDA ABI, or GPU architecture.

```bash
git submodule update --init third_party/DCVC
export PYTHON_BIN=/path/to/native/python
export WEIGHTS=/path/to/official/checkpoints
export CUDA_HOME=/path/to/cuda-toolkit
export CUTLASS_HOME=/path/to/cutlass-v4.4.1
bash scripts/hla_tok/uf/build_probe.sh
bash scripts/hla_tok/uf/build_rans.sh
export PYTHONPATH=.build/uf/probe/site:.build/uf/native:.
export UF_OFFICIAL_WEIGHTS="$WEIGHTS"
"$PYTHON_BIN" -m pytest tests/hla_tok/uf
"$PYTHON_BIN" -m scripts.hla_tok.uf.history verify
```

For independent measurements, run each command on the GPU serially. The `baseline/isolated` tree contains the accepted allocator-isolated baseline; the pre-fix Anchor summaries are retained as failure evidence, not as input to model selection.

```bash
OUT=results/hla_tok/uf
BASE="$OUT/baseline/isolated"
LOCK="$BASE/base_lock.json"
CK=checkpoints/hla_tok/uf/isolated
TRAIN="$OUT/anchor/isolated-training-seed1"
"$PYTHON_BIN" -m scripts.hla_tok.uf.baseline calibration --weights "$WEIGHTS" --output "$BASE"
"$PYTHON_BIN" -m scripts.hla_tok.uf.baseline validation --weights "$WEIGHTS" --output "$BASE"
"$PYTHON_BIN" -m scripts.hla_tok.uf.native_audit --weights "$WEIGHTS" --output "$BASE/native_latents.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf.anchor_eval --weights "$WEIGHTS" --base-lock "$LOCK" \
  --split calibration --output "$OUT/anchor/isolated-A0-calibration"
"$PYTHON_BIN" -m scripts.hla_tok.uf.train_anchor --weights "$WEIGHTS" --base-lock "$LOCK" \
  --steps 500 --checkpoints "$CK" --output "$TRAIN"
"$PYTHON_BIN" -m scripts.hla_tok.uf.anchor_eval --weights "$WEIGHTS" --base-lock "$LOCK" \
  --checkpoint "$CK/anchor-seed1-step500.pt" --split calibration --output "$OUT/anchor/isolated-A1-calibration"
"$PYTHON_BIN" -m scripts.hla_tok.uf.pilot --calibration "$OUT/anchor/isolated-A1-calibration" \
  --training "$TRAIN" --native-audit "$OUT/evidence/native_process_parity.json" \
  --output "$OUT/anchor/A2_admission.json"
```

Only if the native contracts and pilot admission pass:

```bash
"$PYTHON_BIN" -m scripts.hla_tok.uf.train_anchor --weights "$WEIGHTS" --base-lock "$LOCK" \
  --steps 2000 --resume "$CK/anchor-seed1-step500.pt" --checkpoints "$CK" --output "$TRAIN"
for precision in bf16 fp32; do
  "$PYTHON_BIN" -m scripts.hla_tok.uf.anchor_eval --weights "$WEIGHTS" --base-lock "$LOCK" \
    --checkpoint "$CK/anchor-seed1-step2000.pt" --precision "$precision" \
    --output "$OUT/anchor/A2-validation-$precision"
done
"$PYTHON_BIN" -m scripts.hla_tok.uf.unified_eval --weights "$WEIGHTS" --base-lock "$LOCK" \
  --output "$OUT/progressive/D0-validation"
"$PYTHON_BIN" -m scripts.hla_tok.uf.evidence "$OUT/anchor/A2-validation-bf16" \
  --official "$BASE/validation.json" --output "$OUT/evidence/A2_comparison.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf.dynamic --weights "$WEIGHTS" --base-lock "$LOCK" \
  --checkpoint "$CK/anchor-seed1-step2000.pt" --output "$OUT/benchmarks/dynamic-A2" --native-images
for variant in no-ranking static-cdf; do
  "$PYTHON_BIN" -m scripts.hla_tok.uf.unified_eval --weights "$WEIGHTS" --base-lock "$LOCK" \
    --"$variant" --output "$OUT/progressive/D0-$variant"
done
"$PYTHON_BIN" -m scripts.hla_tok.uf.evidence "$OUT/progressive/D0-validation" \
  --official "$BASE/validation.json" --output "$OUT/evidence/D0_comparison.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf.entropy_ablation --samples "$OUT/benchmarks/dynamic-A2/samples" \
  --checkpoint "$CK/anchor-seed1-step2000.pt" --output "$OUT/evidence/static_cdf_same_symbols.json"
"$PYTHON_BIN" -m scripts.hla_tok.uf.plot_evidence --comparison "$OUT/evidence/A2_comparison.json" \
  --anchor "$OUT/anchor/A2-validation-bf16/summary.json" \
  --unified "$OUT/progressive/D0-validation/summary.json" --output "$OUT/figures"
"$PYTHON_BIN" -m scripts.hla_tok.uf.plot_diagnostics --results "$OUT" --output "$OUT/figures"
```

For Seed 2 use the same recipe with `--seed 2` in both training calls, `TRAIN="$OUT/anchor/isolated-training-seed2"`, checkpoints named `anchor-seed2-step*.pt`, the calibration directory `isolated-A1-seed2-calibration` and its own admission `A2_seed2_admission.json`; validation emits `A2-seed2-validation-bf16`. A new A1 produced with the current source can resume with that source. The retained Seed 1 checkpoints used the `e74a79d` source for optimizer restore; its source hash guard deliberately rejects optimizer restore after the recorded codec modules change. They remain readable for evaluation with the recorded model tag. To continue that original optimizer run, use the original source in a separate worktree and verify the checkpoint digest.

The baseline QP grid is `0, 8, 16, 24, 32, 40, 48, 56, 63`. B3 is evaluated with GOP 32 and the upstream initial-I/periodic-feature-reset schedule, B4 is every frame native Intra. Actual rates use 30 FPS and the measured application bytes. Timing includes final full-family entropy and application packing for all bandwidths, plus one receiver's actual decode/render. It excludes disk, display hardware, network, reference-comparison copies, and image quality metrics. PyTorch stays eager; four frames warm each harness, and startup latency is outside these distributions. The dynamic harness keeps the full native PNG set only when explicitly requested; never commit those input pixels.

Test must remain sealed if any final admission constraint fails. These commands only use frozen Train, Calibration, and Validation plans, including the long Validation subset. A formal Test command is intentionally absent for this run.
