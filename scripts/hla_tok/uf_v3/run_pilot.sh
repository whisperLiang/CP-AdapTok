#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the CUDA environment Python}"
v2="${UFV2_HISTORY:?Set UFV2_HISTORY to the pinned V2 checkout}"
weights="${UF_WEIGHTS:?Set UF_WEIGHTS to the official checkpoint directory}"
out="${UFV3_OUT:-results/hla_tok/uf_v3}"
checkpoints="${UFV3_CHECKPOINTS:-checkpoints/hla_tok/uf_v3}"
export PYTHONPATH=".build/uf_v3/native:.build/uf_v3/records:.build/uf_v3/legacy:.${PYTHONPATH:+:$PYTHONPATH}"
export OMP_WAIT_POLICY=PASSIVE

for file in diagnostics/{capacity,validation/capacity}.json entropy/summary.json audit/training_audit.json; do
  test -s "$out/$file"
done
init="$v2/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"
fit=(--weights "$weights" --init "$init" --checkpoints "$checkpoints" --decision "$out/diagnostics/capacity.json"
     --entropy "$out/entropy/summary.json")
backend="$("$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1]))["default"])' "$out/entropy/summary.json")"
static=()
if [[ "$backend" != "E0" ]]; then static+=(--static "$out/entropy/static_cdf.json"); fi
if [[ "$backend" == "E2" ]]; then static+=(--raw); fi

fit_node() {
  local stage="$1"
  local stop="$2"
  shift 2
  local sample_dir="$out/training/$stage-$stop"
  if [[ ! -s "$sample_dir/summary.json" ]]; then
    "$python_bin" -u -m scripts.hla_tok.uf_v3.train "${fit[@]}" --stage "$stage" --steps "$stop" \
      --output "$sample_dir" "$@"
  fi
}

pilot() {
  local name="$1"
  local checkpoint="$2"
  local validation="$out/pilot/$name"
  if [[ ! -s "$validation/summary.json" ]]; then
    "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate --weights "$weights" --checkpoint "$checkpoint" \
      --output "$validation" --split calibration --limit 12 --quality 5 10 20 40 "${static[@]}"
  fi
  "$python_bin" -m scripts.hla_tok.uf_v3.pilot_gate --candidate "$validation" \
    --reference "$out/diagnostics/calibration" --output "$validation/gate.json"
}

fit_node P1 2
fit_node P1 500 --resume "$checkpoints/r0-P1-seed1-step2.pt"
pilot P1 "$checkpoints/r0-P1-seed1-step500.pt"
fit_node P2 2000 --resume "$checkpoints/r0-P1-seed1-step500.pt"
pilot P2 "$checkpoints/r0-P2-seed1-step2000.pt"
