#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
primary="${HTS_PROJECT:?set HTS_PROJECT to the original CP-AdapTok checkout}"
history_v2="${HTS_HISTORY_V2:?set HTS_HISTORY_V2 to the pinned UF-V2 checkout}"
history_v3="${HTS_HISTORY_V3:?set HTS_HISTORY_V3 to the pinned UF-V3 checkout}"
python_bin="${PYTHON_BIN:-$primary/.venv-native/bin/python}"
original_python="${ORIGINAL_PYTHON_BIN:-$primary/.venv-adaptok-official/bin/python}"
weights="${HTS_WEIGHTS:-checkpoints/hla_tok/hts/official}"
results="${HTS_RESULTS:-results/hla_tok/hts}"
native="$results/native_audit"
logs="$results/logs"
mkdir -p "$logs"
export CUDA_VISIBLE_DEVICES=0 OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=.build/hts/native/site:.
export CUTLASS_HOME="${CUTLASS_HOME:-$primary/.build/cutlass}"
export CUDA_HOME="${CUDA_HOME:-$primary/.build/cuda-packages/nvidia/cu13}"

logged() {
  local name="$1"
  shift
  local code=0
  printf '%q ' "$@" > "$logs/$name.command"
  printf '\n' >> "$logs/$name.command"
  "$@" > "$logs/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$logs/$name.exit"
  printf '%s %s\n' "$name" "$code"
  return "$code"
}

logged assets "$python_bin" -m scripts.hla_tok.hts.audit_assets \
  --weights "$weights" --previous-weights "$primary/checkpoints" \
  --cutlass "$CUTLASS_HOME" --nvcc "$CUDA_HOME/bin/nvcc" \
  --v2 "$history_v2" --v3 "$history_v3" --output "$results/experiment_manifest.json"
logged native_capacity "$python_bin" -u -m scripts.hla_tok.hts.native_capacity \
  --weights "$weights" --output "$native"
logged original_adaptok "$original_python" -u -m scripts.hla_tok.hts.replay_original \
  --output "$native/legacy/original"

previous_build="$history_v3/.build/uf_v3"
export PYTHONPATH="$previous_build/native:$previous_build/pack-reuse:$previous_build/legacy:."
cdf="$history_v3/results/hla_tok/uf_v3/entropy/static_cdf.json"
a2="$history_v2/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"
a4="$history_v3/checkpoints/hla_tok/uf_v3/r0-probe-seed1-step2001.pt"
budgets=(0 0.5 1 1.39471 2 5 10 20 40)
logged v2_e1 "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate \
  --weights "$weights" --checkpoint "$a2" --static "$cdf" --split calibration \
  --repetitions 0 --quality "${budgets[@]}" --output "$native/legacy/a2"
logged v3_a4 "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate \
  --weights "$weights" --checkpoint "$a4" --static "$cdf" --split calibration \
  --repetitions 0 --quality "${budgets[@]}" --output "$native/legacy/a4"
