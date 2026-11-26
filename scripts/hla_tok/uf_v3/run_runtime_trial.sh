#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the CUDA environment Python}"
weights="${UF_WEIGHTS:?Set UF_WEIGHTS to the official checkpoint directory}"
out="${UFV3_OUT:-results/hla_tok/uf_v3}"
checkpoint="${1:?Pass the fitted checkpoint}"
build=.build/uf_v3/pack-reuse
export OMP_WAIT_POLICY=PASSIVE
mkdir -p "$build"
git show ffbb7177e869b49934bcddf37c24fc7eade8bf41:hla_tok/uf_v2/records.cpp > "$build/records.cpp"
patch --directory="$build" --fuzz=0 < scripts/hla_tok/uf_v3/patches/pack_reuse.patch
suffix="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
read -ra includes <<< "$("$python_bin" -m pybind11 --includes)"
"${CXX:-c++}" -std=c++17 -shared -fPIC -DNDEBUG -O3 -fopenmp "${includes[@]}" "$build/records.cpp" \
  -lz -o "$build/hla_uf_v2_records$suffix"

common=(--weights "$weights" --checkpoint "$checkpoint" --split calibration --limit 32 --repetitions 3
        --demand --timing-budgets 40 --quality 40 --static "$out/entropy/static_cdf.json")
base_path=.build/uf_v3/native:.build/uf_v3/records:.build/uf_v3/legacy:.
PYTHONPATH="$base_path" "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate "${common[@]}" \
  --output "$out/runtime_trial/original"
PYTHONPATH="$build:$base_path" "$python_bin" -u -m scripts.hla_tok.uf_v3.evaluate "${common[@]}" \
  --output "$out/runtime_trial/reuse"
PYTHONPATH="$base_path" "$python_bin" -m scripts.hla_tok.uf_v3.runtime_trial "$out/runtime_trial"
