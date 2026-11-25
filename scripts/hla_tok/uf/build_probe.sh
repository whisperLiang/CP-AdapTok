#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root="$PWD"
python_bin="${PYTHON_BIN:-python}"
toolkit="${CUDA_HOME:?set CUDA_HOME to the matching CUDA toolkit}"
cutlass="${CUTLASS_HOME:?set CUTLASS_HOME to the pinned CUTLASS checkout}"
probe="$root/.build/uf/probe"
src="$probe/src/layers/extensions/inference"
env_path="$($python_bin -c 'import sys; print(sys.prefix)')"
torch_path="$($python_bin -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
suffix="$($python_bin -c 'from sysconfig import get_config_var; print(get_config_var("EXT_SUFFIX"))')"
python_headers="$($python_bin -c 'from sysconfig import get_path; print(get_path("include"))')"
mkdir -p "$probe/src/cpp" "$src" "$probe/third_party" "$probe/site"
test -f "$probe/.source-installed" || {
  cp -a third_party/DCVC/src/cpp/. "$probe/src/cpp/"
  cp -a third_party/DCVC/src/layers/extensions/inference/. "$src/"
  ln -sfn "$cutlass" "$probe/third_party/cutlass"
  patch --directory="$src" -p1 --fuzz=0 < native_patches/decoded_latent.patch
  patch --directory="$src" -p1 --fuzz=0 < native_patches/hla_uf_native_isolation.patch
  patch --directory="$src" -p1 --fuzz=0 < native_patches/hla_uf_probe.patch
  touch "$probe/.source-installed"
}
export CUDA_HOME="$toolkit"
export CPLUS_INCLUDE_PATH="$env_path/lib/python3.12/site-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PATH="$env_path/bin:$PATH"
if [[ -n "${UF_NATIVE_BUILD_CACHE:-}" ]]; then
  cached="$(realpath "$UF_NATIVE_BUILD_CACHE")"
  jobs=()
  replacements=()
  for unit in bind dmci_proxy dmc_ld_proxy; do
    c++ -c -std=c++17 -O3 -DNDEBUG -fPIC -pthread -Wno-deprecated-declarations \
      -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=inference_extensions_cuda \
      -I"$torch_path/include" -I"$torch_path/include/torch/csrc/api/include" -I"$toolkit/include" \
      -I"$probe/src/cpp/py_rans" -I"$cutlass/include" -I"$cutlass/tools/util/include" -I"$python_headers" \
      "$src/$unit.cpp" -o "$probe/$unit.o" &
    jobs+=("$!")
    replacements+=("$probe/$unit.o")
  done
  for job in "${jobs[@]}"; do wait "$job"; done
  mapfile -d '' objects < <(find "$cached" -type f -name '*.o' ! -name 'bind.o' \
    ! -name 'dmci_proxy.o' ! -name 'dmc_ld_proxy.o' -print0 | sort -z)
  test "${#objects[@]}" -ge 10
  c++ -shared "${replacements[@]}" "${objects[@]}" -L"$torch_path/lib" -L"$toolkit/lib64" \
    -lc10 -ltorch -ltorch_cpu -ltorch_python -lc10_cuda -ltorch_cuda -lcudart \
    -o "$probe/site/inference_extensions_cuda$suffix"
else
  (cd "$src"; "$python_bin" setup.py build_ext --build-lib "$probe/site")
fi
