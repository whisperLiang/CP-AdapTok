#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
weights="${UF_OFFICIAL_WEIGHTS:?Set the verified official weights directory}"
python_bin="${PYTHON_BIN:-python}"
output="${UFV2_RESULTS:-results/hla_tok/uf_v2}"
checkpoints="${UFV2_CHECKPOINTS:-checkpoints/hla_tok/uf_v2}"
seed="${1:-1}"
case "$seed" in
  1|2) ;;
  *) exit 2 ;;
esac

baseline="$output/d1/joint-seed$seed-step8000-validation/summary.json"
test -s "$baseline"
nodes=("$baseline")
previous=8000
mkdir -p .build/uf_v2
for step in 10000 12000 14000 16000 18000 20000; do
  target="$output/training/formal-seed$seed"
  result="$output/d1/formal-seed$seed-step$step-validation"
  prefix="$checkpoints/d1-joint-seed$seed-step"
  log=".build/uf_v2/formal-seed$seed-$step"
  if [[ ! -s "$target/step-$step.json" || ! -s "$prefix$step.pt" ]]; then
    "$python_bin" -m scripts.hla_tok.uf_v2.train_formal --weights "$weights" --seed "$seed" \
      --resume "$prefix$previous.pt" --steps "$step" --checkpoints "$checkpoints" --output "$target" \
      > "$log-training.log" 2>&1
  fi
  if [[ ! -s "$result/summary.json" ]]; then
    "$python_bin" -m scripts.hla_tok.uf_v2.evaluate --weights "$weights" --method d1 \
      --checkpoint "$prefix$step.pt" --repetitions 0 --quality 5 10 20 40 --output "$result" \
      > "$log-validation.log" 2>&1
  fi
  nodes+=("$result/summary.json")
  decision=$("$python_bin" -m scripts.hla_tok.uf_v2.continuation "${nodes[@]}" \
    --output "$target/continuation.json")
  printf '%s %s\n' "$step" "$decision"
  if [[ "$decision" != CONTINUE ]]; then
    break
  fi
  previous=$step
done
