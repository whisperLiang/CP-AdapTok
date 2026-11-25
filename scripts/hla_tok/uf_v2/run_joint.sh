#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
weights="${UF_OFFICIAL_WEIGHTS:?Set the verified official weights directory}"
python_bin="${PYTHON_BIN:-python}"
out="${UFV2_RESULTS:-results/hla_tok/uf_v2}"
ck="${UFV2_CHECKPOINTS:-checkpoints/hla_tok/uf_v2}"
seed="${1:-1}"
case "$seed" in
  1|2) ;;
  *) exit 2 ;;
esac

target="$out/training/joint-seed$seed"
prefix="$ck/d1-joint-seed$seed-step"
previous="$ck/d1-seed$seed-step2000.pt"
mkdir -p .build/uf_v2
for step in 4000 6000 8000; do
  node="$prefix$step.pt"
  evidence="$out/d1/joint-seed$seed-step$step-validation"
  log=".build/uf_v2/joint-seed$seed-$step"
  if [[ ! -s "$target/step-$step.json" || ! -s "$node" ]]; then
    "$python_bin" -m scripts.hla_tok.uf_v2.train_joint --weights "$weights" --seed "$seed" \
      --resume "$previous" --steps "$step" --checkpoints "$ck" --output "$target" > "$log-training.log" 2>&1
  fi
  if [[ ! -s "$evidence/summary.json" ]]; then
    "$python_bin" -m scripts.hla_tok.uf_v2.evaluate --weights "$weights" --method d1 \
      --checkpoint "$node" --repetitions 0 --quality 5 10 20 40 --output "$evidence" > "$log-validation.log" 2>&1
  fi
  printf '%s\n' "$step"
  previous="$node"
done
