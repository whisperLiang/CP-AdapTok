#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the CUDA environment Python}"
v2="${UFV2_HISTORY:?Set UFV2_HISTORY to the pinned V2 checkout}"
weights="${UF_WEIGHTS:?Set UF_WEIGHTS to the verified official weight directory}"
out="${UFV3_OUT:-results/hla_tok/uf_v3}"
fit="${1:?Pass the fitted checkpoint}"
export OMP_WAIT_POLICY=PASSIVE
base_path=.build/uf_v3/native:.build/uf_v3/records:.build/uf_v3/legacy:.
reuse=".build/uf_v3/pack-reuse:$base_path"
suffix="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
binary=".build/uf_v3/pack-reuse/hla_uf_v2_records$suffix"
test -s "$binary"
common=(--weights "$weights" --checkpoint "$fit" --static "$out/entropy/static_cdf.json")
for mode in full_max on_demand; do
  target="$out/runtime_trial/validation_$mode"
  if [[ -s "$target/summary.json" ]]; then continue; fi
  flags=()
  if [[ "$mode" == "on_demand" ]]; then flags+=(--demand); fi
  PYTHONPATH="$reuse" "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate "${common[@]}" \
    "${flags[@]}" --split validation --repetitions 3 --output "$target"
done
for backend in original reuse; do
  filename="$out/runtime_trial/validation_${backend}_family.json"
  if [[ -s "$filename" ]]; then continue; fi
  tool_path="$base_path"
  if [[ "$backend" == "reuse" ]]; then tool_path="$reuse"; fi
  PYTHONPATH="$tool_path" "$python_bin" -m scripts.hla_tok.uf_v3.family_ledger "${common[@]}" --output "$filename"
done
target="$out/runtime_trial/reuse_contracts"
if [[ ! -s "$target/summary.json" ]]; then
  PYTHONPATH="$reuse" "$python_bin" -u -m scripts.hla_tok.uf_v3.contracts "${common[@]}" \
    --init "$v2/checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt" --output "$target"
fi
PYTHONPATH="$reuse" "$python_bin" -m scripts.hla_tok.uf_v3.select_runtime "$out" "$binary"
