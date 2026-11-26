#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the CUDA environment Python}"
v2="${UFV2_HISTORY:?Set UFV2_HISTORY to the pinned V2 checkout}"
weights="${UF_WEIGHTS:?Set UF_WEIGHTS to the official checkpoint directory}"
out="${UFV3_OUT:-results/hla_tok/uf_v3}"
checkpoint="${1:?Pass the fitted V3 checkpoint}"
record_build="${UFV3_RECORDS_DIR:-.build/uf_v3/records}"
export PYTHONPATH=".build/uf_v3/native:$record_build:.build/uf_v3/legacy:.${PYTHONPATH:+:$PYTHONPATH}"
export OMP_WAIT_POLICY=PASSIVE
init="$v2/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"
backend="$("$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1]))["default"])' "$out/entropy/summary.json")"
static=()
if [[ "$backend" != "E0" ]]; then static+=(--static "$out/entropy/static_cdf.json"); fi
if [[ "$backend" == "E2" ]]; then static+=(--raw); fi
common=(--weights "$weights" --checkpoint "$checkpoint" "${static[@]}")
eval=(--split validation --repetitions 3)
if [[ ! -s "$out/evaluation/A4/summary.json" ]]; then
  "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate "${common[@]}" "${eval[@]}" --output "$out/evaluation/A4" \
    --quality 0 0.5 1 1.3947 1.395 2 3 3.594 4 5 10 20 40
fi
if [[ ! -s "$out/evaluation/on_demand/summary.json" ]]; then
  "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate "${common[@]}" "${eval[@]}" --demand --output "$out/evaluation/on_demand"
fi
if [[ ! -s "$out/evaluation/contracts/summary.json" ]]; then
  "$python_bin" -u -m scripts.hla_tok.uf_v3.contracts "${common[@]}" --init "$init" --output "$out/evaluation/contracts"
fi
if [[ ! -s "$out/samples/native/manifest.json" ]]; then
  "$python_bin" -u -m scripts.hla_tok.uf_v3.samples "${common[@]}" --output "$out/samples/native"
fi
"$python_bin" -m scripts.hla_tok.uf_v3.finalize "$out" --checkpoint "$checkpoint"
