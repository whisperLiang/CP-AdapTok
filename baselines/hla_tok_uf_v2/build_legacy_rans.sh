#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin="${PYTHON_BIN:-python}"
upstream=third_party/DCVC/src/cpp/py_rans
output=.build/uf_v2/legacy
mkdir -p "$output"
target="$output/hla_uf_rans$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
read -ra includes <<< "$("$python_bin" -m pybind11 --includes)"
"${CXX:-c++}" -std=c++17 -shared -fPIC -pthread -DNDEBUG -O3 "${includes[@]}" "-I$upstream" \
  baselines/hla_tok_uf_v2/rans_bind.cpp "$upstream/rans.cpp" -o "$target.tmp"
mv -- "$target.tmp" "$target"
