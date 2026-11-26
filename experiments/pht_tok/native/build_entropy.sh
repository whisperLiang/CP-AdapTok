#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python_bin="${PYTHON_BIN:-python}"
site="$PWD/.build/pht-tok/pht/site"
torch_dir="$("$python_bin" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
headers="$("$python_bin" -c 'from sysconfig import get_path; print(get_path("include"))')"
suffix="$("$python_bin" -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
mkdir -p "$site"
c++ -std=c++17 -O3 -fPIC -shared -I"$torch_dir/include" -I"$headers" \
  experiments/pht_tok/native/entropy.cpp -o "$site/pht_entropy_native$suffix"
