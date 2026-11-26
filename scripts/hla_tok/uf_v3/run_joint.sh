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

status() {
  "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

init="$v2/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"
fit=(--weights "$weights" --init "$init" --checkpoints "$checkpoints"
     --decision "$out/diagnostics/capacity.json" --entropy "$out/entropy/summary.json")
backend="$("$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1]))["default"])' "$out/entropy/summary.json")"
static=()
if [[ "$backend" != "E0" ]]; then static+=(--static "$out/entropy/static_cdf.json"); fi
if [[ "$backend" == "E2" ]]; then static+=(--raw); fi

evaluate() {
  local destination="$1"
  local checkpoint="$2"
  shift 2
  if [[ ! -s "$destination/summary.json" ]]; then
    "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate --weights "$weights" --checkpoint "$checkpoint" \
      --output "$destination" --quality 5 10 20 40 "${static[@]}" "$@"
  fi
}

fit_node() {
  local stage="$1"
  local stop="$2"
  local resume="$3"
  if [[ ! -s "$out/training/$stage-$stop/summary.json" ]]; then
    "$python_bin" -u -m scripts.hla_tok.uf_v3.train "${fit[@]}" --stage "$stage" --steps "$stop" \
      --resume "$resume" --output "$out/training/$stage-$stop"
  fi
}

evaluate "$out/evaluation/A2" "$init" --split validation
latest="$checkpoints/r0-P2-seed1-step2000.pt"
if [[ "$(status "$out/pilot/P2/gate.json")" != "PASS" ]]; then
  fit_node probe 2064 "$latest"
  latest="$checkpoints/r0-probe-seed1-step2064.pt"
  evaluate "$out/pilot/probe" "$latest" --split calibration --limit 12
  "$python_bin" -m scripts.hla_tok.uf_v3.pilot_gate --candidate "$out/pilot/probe" \
    --reference "$out/diagnostics/calibration" --output "$out/pilot/probe/gate.json"
  if [[ "$(status "$out/pilot/probe/gate.json")" != "PASS" ]]; then
    printf '%s\n' "$latest" > "$out/training/candidate.txt"
    exit 0
  fi
fi

for node in 4000 6000 8000; do
  fit_node P3 "$node" "$latest"
  latest="$checkpoints/r0-P3-seed1-step$node.pt"
  evaluate "$out/evaluation/step-$node" "$latest" --split validation --repetitions 0
  "$python_bin" -m scripts.hla_tok.uf_v3.formal_gate "$out"
  if [[ "$(status "$out/training/continuation.json")" == "STOP" ]]; then break; fi
done
if [[ "$(status "$out/training/continuation.json")" == "CONTINUE" ]]; then
  for node in 10000 12000 14000 16000 18000 20000; do
    fit_node P4 "$node" "$latest"
    latest="$checkpoints/r0-P4-seed1-step$node.pt"
    evaluate "$out/evaluation/step-$node" "$latest" --split validation --repetitions 0
    "$python_bin" -m scripts.hla_tok.uf_v3.formal_gate "$out"
    if [[ "$(status "$out/training/continuation.json")" == "STOP" ]]; then break; fi
  done
fi
best="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_step"])' "$out/training/continuation.json")"
kind=P3
if (( best > 8000 )); then kind=P4; fi
printf '%s\n' "$checkpoints/r0-$kind-seed1-step$best.pt" > "$out/training/candidate.txt"
