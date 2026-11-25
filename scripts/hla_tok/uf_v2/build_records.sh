#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python_bin="${PYTHON_BIN:-python}"
output="${UFV2_RECORD_BUILD_DIR:-.build/uf_v2/records}"
mkdir -p "$output"
target="$output/hla_uf_v2_records$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
read -ra includes <<< "$("$python_bin" -m pybind11 --includes)"
defines=()
if [[ "${UFV2_PROFILE_CANVAS:-0}" == 1 ]]; then
  defines=(-DUFV2_PROFILE_CANVAS)
fi
"${CXX:-c++}" -std=c++17 -shared -fPIC -DNDEBUG -O3 -fopenmp "${includes[@]}" "${defines[@]}" \
  hla_tok/uf_v2/records.cpp -lz -o "$target.tmp"
mv -- "$target.tmp" "$target"
